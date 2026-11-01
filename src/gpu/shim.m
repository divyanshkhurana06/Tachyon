// A thin C interface over Metal, so Rust can drive the GPU without any
// Objective-C bindings crate: create a device and pipelines, allocate shared
// buffers, record kernel dispatches into one command buffer, run it.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

typedef struct {
    void *buf;
    uint64_t offset;
} TmArg;

@interface TmContext : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) NSMutableArray<id<MTLComputePipelineState>> *pipelines;
@property(strong) id<MTLCommandBuffer> cmd;
@property(strong) id<MTLComputeCommandEncoder> enc;
@end

@implementation TmContext
@end

static void set_err(char *err, size_t len, NSString *msg) {
    if (err && len) {
        const char *s = msg.UTF8String ? msg.UTF8String : "unknown Metal error";
        strncpy(err, s, len - 1);
        err[len - 1] = 0;
    }
}

void *tm_create(const char *source, const char **kernels, size_t n, char *err, size_t errlen) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            // Command-line tools sometimes get no default device; ask for all.
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            dev = all.firstObject;
        }
        if (!dev) {
            set_err(err, errlen, @"no Metal device");
            return NULL;
        }
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&e];
        if (!lib) {
            set_err(err, errlen, e.localizedDescription);
            return NULL;
        }
        TmContext *ctx = [TmContext new];
        ctx.device = dev;
        ctx.queue = [dev newCommandQueue];
        ctx.pipelines = [NSMutableArray new];
        for (size_t i = 0; i < n; i++) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:kernels[i]]];
            if (!f) {
                set_err(err, errlen, [NSString stringWithFormat:@"kernel %s not found", kernels[i]]);
                return NULL;
            }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
            if (!p) {
                set_err(err, errlen, e.localizedDescription);
                return NULL;
            }
            [ctx.pipelines addObject:p];
        }
        return (__bridge_retained void *)ctx;
    }
}

void tm_destroy(void *ctx) {
    if (ctx) {
        TmContext *c = (__bridge_transfer TmContext *)ctx;
        c = nil;
    }
}

void tm_device_name(void *ctx, char *out, size_t len) {
    @autoreleasepool {
        TmContext *c = (__bridge TmContext *)ctx;
        set_err(out, len, c.device.name);
    }
}

void *tm_buffer(void *ctx, size_t len) {
    @autoreleasepool {
        TmContext *c = (__bridge TmContext *)ctx;
        id<MTLBuffer> b = [c.device newBufferWithLength:(len ? len : 16) options:MTLResourceStorageModeShared];
        return b ? (__bridge_retained void *)b : NULL;
    }
}

void *tm_buffer_contents(void *buf) {
    return [(__bridge id<MTLBuffer>)buf contents];
}

void tm_buffer_free(void *buf) {
    if (buf) {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf;
        b = nil;
    }
}

void tm_begin(void *ctx) {
    @autoreleasepool {
        TmContext *c = (__bridge TmContext *)ctx;
        // Unretained: the caller keeps every buffer alive until tm_end returns.
        c.cmd = [c.queue commandBufferWithUnretainedReferences];
        c.enc = [c.cmd computeCommandEncoder];
    }
}

void tm_dispatch(void *ctx, uint32_t pipeline, const TmArg *args, uint32_t nargs, const void *params, uint32_t plen,
                 uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
    @autoreleasepool {
        TmContext *c = (__bridge TmContext *)ctx;
        id<MTLComputeCommandEncoder> enc = c.enc;
        [enc setComputePipelineState:c.pipelines[pipeline]];
        for (uint32_t i = 0; i < nargs; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>)args[i].buf offset:args[i].offset atIndex:i];
        }
        if (plen) {
            [enc setBytes:params length:plen atIndex:nargs];
        }
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    }
}

int tm_end(void *ctx, char *err, size_t errlen) {
    @autoreleasepool {
        TmContext *c = (__bridge TmContext *)ctx;
        [c.enc endEncoding];
        [c.cmd commit];
        [c.cmd waitUntilCompleted];
        int rc = 0;
        if (c.cmd.error) {
            set_err(err, errlen, c.cmd.error.localizedDescription);
            rc = -1;
        }
        c.enc = nil;
        c.cmd = nil;
        return rc;
    }
}
