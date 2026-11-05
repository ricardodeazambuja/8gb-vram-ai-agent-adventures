#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
// H2D bandwidth: pinned vs pageable, sizes 1.4 MiB (one expert) .. 64 MiB; plus RAM read bw with N threads; plus concurrent.
static double h2d(void* dst, void* src, size_t n, int iters){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice);
  cudaEventRecord(a); for(int i=0;i<iters;i++) cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); return (double)n*iters/(ms*1e-3)/1e9;
}
static std::atomic<bool> stop{false};
static double ram_read(const float* p, size_t n, int threads, double secs){
  std::vector<std::thread> th; std::vector<double> bytes(threads,0); std::vector<float> sink(threads);
  auto t0=std::chrono::steady_clock::now();
  for(int t=0;t<threads;t++) th.emplace_back([&,t]{ size_t chunk=n/threads; const float* q=p+t*chunk; double by=0; float s=0;
    while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<secs){ for(size_t i=0;i<chunk;i+=16) s+=q[i]+q[i+4]+q[i+8]+q[i+12]; by+=chunk*4.0;} bytes[t]=by; sink[t]=s;});
  for(auto&x:th) x.join(); double tot=0; for(double b:bytes) tot+=b;
  return tot/std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/1e9;
}
int main(){
  size_t maxn=64<<20; void *d; cudaMalloc(&d,maxn);
  void* pin; cudaMallocHost(&pin,maxn); memset(pin,1,maxn);
  void* pg=malloc(maxn); memset(pg,1,maxn);
  size_t sizes[]={(size_t)(1.412*(1<<20)),(size_t)(3.19*(1<<20)),8<<20,64<<20};
  for(size_t n: sizes){ printf("H2D %7.2f MiB: pinned %5.2f GB/s  pageable %5.2f GB/s\n", n/1048576.0, h2d(d,pin,n,50), h2d(d,pg,n,50)); }
  size_t rn=(size_t)1<<30; float* r=(float*)malloc(rn*4); for(size_t i=0;i<rn;i++) r[i]=1.f;
  for(int t: {1,2,4,6,12}) printf("RAM read %2d threads: %5.1f GB/s\n", t, ram_read(r,rn,t,2.0));
  // concurrent: pinned H2D loop in a thread while 6 threads read RAM
  std::atomic<bool> go{true}; double h2dbw=0;
  std::thread cp([&]{ cudaSetDevice(0); double s=0; int k=0; auto t0=std::chrono::steady_clock::now(); while(go){ s+=h2d(d,pin,64<<20,10); k++; } h2dbw=s/k; });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  double rb=ram_read(r,rn,6,3.0); go=false; cp.join();
  printf("concurrent: RAM read 6 thr %5.1f GB/s + pinned H2D %5.2f GB/s\n", rb, h2dbw);
}
