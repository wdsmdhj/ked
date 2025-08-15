K3pccA片视频51在线观看潍坊军训跳舞动图

不可分页主机内存的分配可以由以下两个 CUDA 运行时 API 函数中的任何一个实现：

cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, size_t flags);

注意，第二个函数的名字中没有字母 M。若函数 cudaHostAlloc 的第三个参数取默认
值 cudaHostAllocDefault，则以上两个函数完全等价。由以上函数分配的主机内存必须由如下函数释放：cudaError_t cudaFreeHost(void* ptr);如果不小心用了 free 函数释放不可分页主机内存，会出现运行错误。
4.2  重叠核函数执行与数据传输的例子

        我们说过，在编写 CUDA 程序时要尽量避免主机与设备之间的数据传输，但这种数据传输一般来说是无法完全避免的。假如在一段 CUDA 程序中，我们需要先从主机向设备传输一定数量的数据（我们将此 CUDA 操作简称为 H2D），然后在 GPU 中使用所传输的数据做一些计算（我们将此 CUDA 操作简称为 KER，意为核函数执行），最后将一些数据从设备传输至主机（我们将此 CUDA 操作简称为 D2H）。下面，我们首先从理论上分析使用 CUDA 流可能带来的性能提升。

        要利用多个流提升性能，就必须创造出在逻辑上可以并发执行的 CUDA 操作。一个方
法是将以上 3 个 CUDA 操作都分成若干等份，然后在每个流中发布一个 CUDA 操作序列。例如，使用两个流时，我们将以上 3 个 CUDA 操作都分成两等份。在理想情况下，它们的执行流程可以如下：

Stream 1：H2D -> KER -> D2H
Stream 2： H2D -> KER -> D2H
