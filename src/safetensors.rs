//! Reader for the safetensors format: an 8-byte little-endian header length,
//! a JSON header giving each tensor's dtype, shape and byte range, then the
//! raw tensor data.
//!
//! Files are memory-mapped, so opening a multi-gigabyte checkpoint is
//! instant and the OS pages in only what is read. Without `mmap` (the
//! browser build) a checkpoint is read from bytes in memory instead.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::kernels::{bf16_to_f32, f16_to_f32};
use crate::{bail, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }
}

pub struct Tensor<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl Tensor<'_> {
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::Bf16 => self
                .data
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        }
    }
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    file: usize,
    start: usize,
    end: usize,
}

pub struct SafeTensors {
    files: Vec<Storage>,
    entries: HashMap<String, Entry>,
}

/// A checkpoint file's bytes: mapped from disk, or owned.
enum Storage {
    #[cfg(unix)]
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            #[cfg(unix)]
            Storage::Mapped(m) => m.bytes(),
            Storage::Owned(v) => v,
        }
    }
}

impl SafeTensors {
    /// Opens every `.safetensors` file in `dir` (large models are sharded).
    pub fn open_dir(dir: &Path) -> Result<SafeTensors> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| Error(format!("cannot read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut st = SafeTensors {
            files: Vec::new(),
            entries: HashMap::new(),
        };
        for path in paths {
            #[cfg(unix)]
            let storage = Storage::Mapped(Mmap::open(&path)?);
            #[cfg(not(unix))]
            let storage = Storage::Owned(fs::read(&path)?);
            st.add(storage, &path.display().to_string())?;
        }
        Ok(st)
    }

    /// A checkpoint already in memory (one file's bytes).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<SafeTensors> {
        let mut st = SafeTensors {
            files: Vec::new(),
            entries: HashMap::new(),
        };
        st.add(Storage::Owned(bytes), "model.safetensors")?;
        Ok(st)
    }

    fn add(&mut self, storage: Storage, ctx: &str) -> Result<()> {
        let bytes = storage.bytes();
        if bytes.len() < 8 {
            bail!("{ctx}: too short to be safetensors");
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = bytes
            .get(8..8usize.saturating_add(n))
            .ok_or_else(|| Error(format!("{ctx}: truncated header")))?;
        let header = std::str::from_utf8(header)
            .map_err(|_| Error(format!("{ctx}: header is not UTF-8")))?;
        let header = Json::parse(header).map_err(|e| Error(format!("{ctx}: {e}")))?;
        let base = 8 + n;
        let file = self.files.len();

        for (name, info) in header
            .as_object()
            .ok_or_else(|| Error(format!("{ctx}: header is not an object")))?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match info.get("dtype").and_then(Json::as_str) {
                Some("F32") => Dtype::F32,
                Some("F16") => Dtype::F16,
                Some("BF16") => Dtype::Bf16,
                other => bail!("{name}: unsupported dtype {other:?}"),
            };
            let shape = info
                .get("shape")
                .and_then(Json::as_array)
                .and_then(|dims| dims.iter().map(Json::as_usize).collect::<Option<Vec<_>>>())
                .ok_or_else(|| Error(format!("{name}: bad shape")))?;
            let offsets = info
                .get("data_offsets")
                .and_then(Json::as_array)
                .and_then(|o| Some((o.first()?.as_usize()?, o.get(1)?.as_usize()?)))
                .ok_or_else(|| Error(format!("{name}: bad data_offsets")))?;
            let (start, end) = (base + offsets.0, base + offsets.1);
            if start > end || end > bytes.len() {
                bail!("{name}: data lies outside the file");
            }
            if end - start != shape.iter().product::<usize>() * dtype.size() {
                bail!("{name}: byte size does not match shape {shape:?}");
            }
            self.entries.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    file,
                    start,
                    end,
                },
            );
        }
        self.files.push(storage);
        Ok(())
    }

    /// Every tensor's name, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error(format!("checkpoint has no tensor {name}")))?;
        Ok(Tensor {
            dtype: e.dtype,
            shape: &e.shape,
            data: &self.files[e.file].bytes()[e.start..e.end],
        })
    }
}

#[cfg(unix)]
use mmap::Mmap;

#[cfg(unix)]
mod mmap {
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::os::fd::AsRawFd;
    use std::path::Path;

    use crate::{bail, Error, Result};

    pub struct Mmap {
        ptr: *mut c_void,
        len: usize,
    }

    // The mapping is read-only and lives as long as the struct.
    unsafe impl Send for Mmap {}
    unsafe impl Sync for Mmap {}

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;

    impl Mmap {
        pub fn open(path: &Path) -> Result<Mmap> {
            let file = File::open(path)
                .map_err(|e| Error(format!("cannot open {}: {e}", path.display())))?;
            let len = file.metadata()?.len() as usize;
            if len == 0 {
                bail!("{} is empty", path.display());
            }
            // The mapping stays valid after `file` is closed.
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize == -1 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(Mmap { ptr, len })
        }

        pub fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            unsafe { munmap(self.ptr, self.len) };
        }
    }
}
