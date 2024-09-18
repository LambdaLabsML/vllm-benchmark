
# LLMs inference benchmarks, using vLLM, on Lambda Cloud

[OpenAI's introduction of O1](https://openai.com/index/introducing-openai-o1-preview/), its latest large language model, underscores how performance can be drastically enhanced through [inference time scaling](https://openai.com/index/learning-to-reason-with-llms/): the longer the model "thinks" during inference, the better it performs in reasoning tasks. This shift is redefining the landscape of LLM research and fueling the growing demand for more efficient inference solutions.

In this report, we benchmark LLM inference across NVIDIA A100 and H100 GPUs, using the popular [vLLM](https://github.com/vllm-project/vllm) framework to test models like [Llama](https://www.llama.com/) and [Mistral](https://docs.mistral.ai/getting-started/models/), ranging from 7B to 405B parameters. Key findings include:

* Despite the trade-off between throughput and latency, faster GPUs and parallelism improve both.
* Data parallelism outperforms tensor parallelism when the model fits within the system.
* NVIDIA H100 offers ~`2x` better throughput and lower latency than A100.
* vLLM v0.6.0 doubles output throughput and reduces Time Per Output Token (TPOT).

You can reproduce the benchmark with [this guide](https://github.com/LambdaLabsML/vllm-benchmark/blob/main/README.md). For users interested in self-hosting Llama 3.1 with vLLM, follow [this tutorial](https://docs.lambdalabs.com/on-demand-cloud/how-to-serve-the-llama-3.1-8b-and-70b-models-using-lambda-cloud-on-demand-instances?utm_source=linkedin&utm_medium=organic-social&utm_campaign=2024-09-vLLM-Benchmark-Report&utm_content=post-a) to deploy the model on [Lambda's Public Cloud](https://lambdalabs.com/service/gpu-cloud).

## Benchmark Design

Here is the summary of all variables in our benchmarks:

__Models__: We benchmark models from the Llama and Mistral families. The model sizes range between 7B to 405B. The full list of models can be found in [this script](https://github.com/LambdaLabsML/vllm-benchmark/blob/main/cache_model.py#L3). 

__GPUs__: We have benchmarked NVIDIA H100-80GB-SXM and NVIDIA A100-80GB-SXM GPUs, in `1x`, `2x`, `4x` and `8x` settings. The reason for not going beyond a single node is the biggest model (`Llama3.1 405B`) can be served by 8x GPUs, hence one can keep tensor parallelism within a single server and scale out the performance more effeciently using data parallelism. 

__Benchmark settings__: The [default vLLM settings](https://docs.vllm.ai/en/latest/models/engine_args.html) are generally very good. However we did experimented different values for the following parameters:
* `--max-model-len`: Defines the model context length, which impacts the memory required for KV cache. We test different values to identify the max context length a specific hardware configuration can support a specific model with.
* `--tensor-parallel-size`: Defines how to splits a model across multiple GPUs, so to serve models that are too large to fit on a single GPU.
* `--num-prompts`: Beside vLLM settings, The [benchmark script](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) provides these argument to control the number of requests. Notice vLLM will [automatically batch](https://github.com/vllm-project/vllm/issues/1707#issuecomment-1816797973) these requests in a optimzed way, as long as they are [sent asynchronously](https://github.com/vllm-project/vllm/issues/2257#issuecomment-1869400614) (as implemented in the [benchmark script](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)).

__Metrics__: Our benchmarks monitor `Output token throughput` and `Inter-token Latency`. Both of them are important performance metrics for LLM inference. As we will show later, there is also a trade-off between them: one can increase the overall throughput by increasing the batch size, at the cost of increasing the latency. We captured such a trade off by conducting benchmarks with different values for the aforementioned `--num-prompts` parameter.


## Results

### Max Context Length

The max context length that can be served for a specific model depends on GPU's VRAM size, and number of GPUs that are used for tensor parallelization. For example, a single NVIDIA A100/H100 80GB can serve `Meta-Llama-3.1-70B-Instruct-FP8` with upto 4k context window, wheres 2x NVIDIA A100/H100 80GB can serve the same model upto 128k context length. For the largest Llama Model (405B), one need 8x NVIDIA A100/H100 80GB to serve at its full 128k context length.

| GPU                | Meta-Llama-3.1-8B-FP8 | Meta-Llama-3.1-70B-Instruct-FP8 | Hermes-3-Llama-3.1-405B-FP8 | Mistral-7B-Instruct-v0.3 | Mixtral-8x7B-Instruct-v0.1 | Mixtral-8x22B-Instruct-v0.1 | Mistral-Nemo-Instruct-2407 | Mistral-Large-Instruct-2407 |
|--------------------|-----------------------|---------------------------------|-----------------------------|---------------------------|----------------------------|-----------------------------|----------------------------|-----------------------------|
| 1xA100-80GB-SXM    | 128k                | 4k                            | 0                           | 32k                     | 0                          | 0                           | 128k                    | 0                           |
| 2xA100-80GB-SXM    | 128k                | 128k                          | 0                           | 32k                     | 32k                      | 0                           | 128k                    | 0                           |
| 4xA100-80GB-SXM    | 128k                | 128k                          | 0                           | 32k                     | 32k                      | 16k                       | 128k                    | 128k                     |
| 8xA100-80GB-SXM    | 128k                | 128k                          | 128k                      | 32k                     | 32k                      | 64k                       | 128k                    | 128k                     |
| 1xH100-80GB-SXM    | 128k                | 4k                            | 0                           | 32k                     | 0                          | 0                           | 128k                    | 0                           |
| 2xH100-80GB-SXM    | 128k                | 128k                          | 0                           | 32k                     | 32k                      | 0                           | 128k                    | 0                           |
| 4xH100-80GB-SXM    | 128k                | 128k                          | 0                           | 32k                     | 32k                      | 16k                       | 128k                    | 128k                     |
| 8xH100-80GB-SXM    | 128k                | 128k                          | 128k                      | 32k                     | 32k                      | 64k                       | 128k                    | 128k                     |


### Throughput v.s. Latency curve
One very useful way to understand the performance is through the lens of the "throughput v.s. latency" graph, produced by changing the batch size used for inference. As mentioned earlier, there is usually a trade off between these two metrics: as the batch size increases, inference changes from being memory bandwidth bottlenecked (low latency, low throughput) to being compute bottlenecked (high latency, high throughput). The graph will eventually plateau when all NVIDIA tensor cores are staturated -- adding more data to a batch won't improve the throughput anymore.

Each of the following graphs shows the "throughput v.s. latency" profile for a specific model across different GPUs. The stronger performance profiles are closer to the top left corner (lower latency, higher throughput). Despite both NVIDIA A100 and H100 have the same amount of GPU ram (`80GB`), it is clear that the faster GPU (NVIDIA H100) and more GPUs (enables tensor parallelization) give stronger performance. 

<p align="center">
  <img src="./renders_v0/Mistral-7B-Instruct-v0.3_len2000.png" alt="Mistral-7B-Instruct-v0.3" width="45%" />
  <img src="./renders_v0/Mixtral-8x7B-Instruct-v0.1_len2000.png" alt="Mixtral-8x7B-Instruct-v0.1" width="45%" />
  <!-- <img src="./renders_v0/Mixtral-8x22B-Instruct-v0.1_len2000.png" alt="Mixtral-8x22B-Instruct-v0.1" width="30%" /> -->
</p>


Similarly, we can plot the "throughput v.s. latency" profile for the same GPU but across different models. It is no surprise that given the same GPU, the profile of smaller models are closer to the top left. And in general serving larger models require more GPUs, as some of the models are missing from the 1xH100 figure (left).

<p align="center">
  <img src="./renders_v0/1xH100-80GB-SXM.png" alt="1xH100-80GB-SXM" width="45%" />
  <!-- <img src="./renders_v0/2xH100-80GB-SXM.png" alt="2xH100-80GB-SXM" width="45%" /> -->
  <img src="./renders_v0/8xH100-80GB-SXM.png" alt="8xH100-80GB-SXM" width="45%" />
</p>


### Tensor Parallel v.s. Data Parallel
Which is a better way to scale the performance? Is it better to scale vertically using tensor parallelism, or is it better to scale horizontally with data parallelism. The former gives you a "beefier" processor by combining the memory and tensor cores from multiple GPUs, at the cost of inter-device communication; while the later keep each GPUs independent so you have a fleet of less powerful devices. 

The following figures illustrate the different characteristics of these two parallelism strategies. To do so, we doubled the number of GPUs and applied either DP to horizontally scale the system, or TP to vertically scale the system. We also double the number of prompts used in the system so to make sure data parallelism could double its throughput while keeping the latency unaffected. Our benchmark showed tensor parallelism runs at lower latencies, while data parallelism runs at higher throughputs. 

<p align="center">
  <img src="./renders/scale_Mistral-7B-Instruct-v0.3_len2000.png" alt="scale_Mistral-7B-Instruct-v0.3_len2000" width="45%" />
  <img src="./renders/scale_Mixtral-8x7B-Instruct-v0.1_len2000.png" alt="scale_Mixtral-8x7B-Instruct-v0.1_len2000" width="45%" />
  <!-- <img src="./renders/scale_Mixtral-8x22B-Instruct-v0.1_len2000.png" alt="scale_Mixtral-8x22B-Instruct-v0.1_len2000" width="30%" /> -->
</p>

In general, scaling inference using data parallelism is often more effective than tensor parallelism, if the model fits within the system. The table below provides some data, showing how `throughput/latency` scales poorly for "overly" tensor parallelized systems: with `--num-prompts` fixed at 320 to ensure a large batch size and fully utilize the compute, the throughputs of tensor parallelism still scale far from linearly, unlike the expected behavior with data parallelism.

|                  | Meta-Llama-3.1-8B-FP8 | Meta-Llama-3.1-70B-Instruct-FP8 | Hermes-3-Llama-3.1-405B-FP8 | Mistral-7B-Instruct-v0.3 | Mixtral-8x7B-Instruct-v0.1 | Mixtral-8x22B-Instruct-v0.1 | Mistral-Nemo-Instruct-2407 | Mistral-Large-Instruct-2407 |
|------------------|----------------------:|--------------------------------:|----------------------------:|-------------------------:|---------------------------:|----------------------------:|----------------------------:|-----------------------------:|
| 1xA100-80GB-SXM  | 1517.54/42.85          | 208.79/58.26                    | N/A                         | 1591.4/56.51             | N/A                        | N/A                         | 1277.85/61.48               | N/A                          |
| 2xA100-80GB-SXM  | 1854.27/38.21          | 662.64/120.32                   | N/A                         | 1861.61/53.52            | 1017.34/67.24              | N/A                         | 1593.97/51.39               | N/A                          |
| 4xA100-80GB-SXM  | 1899.76/40.02          | 956.53/88.25                    | N/A                         | 2228.35/44.66            | 1098.11/56.05              | 715.37/95.54                | 1919.13/44.7                | 642.03/129.96                |
| 8xA100-80GB-SXM  | 1972.26/40.47          | 1162.78/74.12                   | 467.85/180.74               | 2381.43/39.89            | 1289.26/49.91              | 946.22/71.43                | 2035.34/40.7                | 947.56/88.72                 |
| 1xH100-80GB-SXM  | 3400.4/19.75           | 418.92/35.84                    | N/A                         | 3281.19/27.23            | N/A                        | N/A                         | 2414.81/31.16               | N/A                          |
| 2xH100-80GB-SXM  | 3688.64/18.83          | 1727.69/49.33                   | N/A                         | 3758.54/24.26            | 1430.42/38.08              | N/A                         | 3064.06/24.44               | N/A                          |
| 4xH100-80GB-SXM  | 3475.27/20.26          | 2242.99/38.26                   | N/A                         | 4133.32/21.79            | 2102.86/30.36              | 1190.38/51.35               | 3509.24/22.1                | 1301.24/56.42                |
| 8xH100-80GB-SXM  | 3931.83/17.61          | 2572.8/33.71                    | 1230.74/73.54               | 4397.11/21.56            | 2073.9/25.88               | 1558.66/39.25               | 3686.29/22.85               | 1734.19/46.23                |


### Performance v.s. max_model_len
Although the `max_model_len` decides the max context length a system can support a model with, it is interesting that the "throughput v.s. latency" profile doesn't vary by it. As shown in the figure below, there is little difference between the benchmark outcomes of the same system using `max_model_len` range between `2000`  to `128000`. 

<p align="center">
  <img src="./renders/1xH100-80GB-SXM_Meta-Llama-3.1-8B-FP8.png" alt="1xH100-80GB-SXM_Meta-Llama-3.1-8B-FP8" width="45%" />
  <img src="./renders/2xH100-80GB-SXM_Meta-Llama-3.1-70B-Instruct-FP8.png" alt="2xH100-80GB-SXM_Meta-Llama-3.1-70B-Instruct-FP8" width="45%" />
  <!-- <img src="./renders/8xH100-80GB-SXM_Hermes-3-Llama-3.1-405B-FP8.png" alt="8xH100-80GB-SXM_Hermes-3-Llama-3.1-405B-FP8" width="30%" /> -->
</p>


### H100 v.s. A100
The performance gap between NVIDIA H100 80GB SXM and A100 80GB SXM varies from model to model. Overall H100 can deliver around 2x higher throughput and 2x lower latency. As an example, for serving `Mistral-7B-Instruct-v0.3`, `1xH100` delivers 2.06x higher througput and 2.07x lower latency. For serving `Hermes-3-Llama-3.1-405B-FP8`, `8xH100` delivers 2.65x higher throughput and 2.45x lower latency. 


### vLLM v0.5.4 v.s. v0.6.0
This benchmark was conducted with vLLM `v0.5.4`. We observed major improvements with the latest version, `v0.6.0`, particularly in reducing CPU overhead. Our tests confirm that `v0.6.0` more than doubles output throughput and reduces Median Time Per Output Token (TPOT) for Llama 3.1 8B and 70B models. This is largely thanks to the multi-step scheduling feature (`num-scheduler-steps=10`). The performance gain is smaller for the Llama 405B model, likely due to its heavier computation where CPU bottlenecks are less significant.

A key improvement in `v0.6.0` is higher GPU utilization. For example, in the Llama 3.1 8B benchmark, GPU power draw increased from `~60%` in `v0.5.4` to `~95%`.

One "tradeoff" is the higher median inter-token latency (ITL) in `v0.6.0`, as [reported by other community contributors](https://github.com/sgl-project/sglang/tree/main/benchmark/benchmark_vllm_060). Since tokens are streamed only after a batch of generation steps is complete, ITL will be inflated by the `num-scheduler-steps`. This can cause some "chunkiness" in streaming, but it’s unlikely to affect user experience since most LLM services stream faster than the human reading speed.

The tables below compare the performance of `v0.5.4`, `v0.6.0+step1`, and `v0.6.0+step10` across three different Llama 3.1 models. The impact of `num-scheduler-steps` is clear, showing significant improvements in output throughput and Median TPOT. We set requests per second (`rps`) to `inf` to simulate high inbound traffic. For all tests, we set `max-num-seq=256` and `max-seq-len=2048`. As of this writing, the vLLM team is working on [PR#8001](https://github.com/vllm-project/vllm/pull/8001) to support `num-scheduler-steps` with chunked prefill, allowing the latest optimizations to run at full context length for large models.

`Llama 3.1 8B`: `tp=1`, `rps=inf`, `num-prompt=5000`, `max-num-seq=256`, `max-seq-len=2048`
| vLLM version | Chuncked Prefill | Scheduler Steps | Output Throughput (tokens/sec) | Median TPOT (ms) | Median ITL (ms) |
|--------------|------------------|-----------------|--------------------------------|------------------|-----------------|
| v0.5.4       | True             | N/A             | 3032.04           | 64.06       | 48.28      | 
| v0.6.0       | False            | 1               | 3958.40           | 196.24      | 179.17     | 
| v0.6.0       | False            | 10              | 8088.35           | 25.57       | 243.76     | 


`Llama 3.1 70B`: `tp=4`, `rps=inf`, `num-prompt=5000`, `max-num-seq=256`, `max-seq-len=2048`
| vLLM version | Chuncked Prefill | Scheduler Steps | Output Throughput (tokens/sec) | Median TPOT (ms) | Median ITL (ms) |
|--------------|------------------|-----------------|--------------------------------|------------------|-----------------|
| v0.5.4       | True             | N/A             | 1491.93           | 125.78      | 74.22      | 
| v0.6.0       | False            | 1               | 2291.0            | 102.99      | 63.54      | 
| v0.6.0       | False            | 10              | 3542.88           | 61.10       | 574.07     | 


`Llama 3.1 405B FP8`: `tp=8`, `rps=inf`, `num-prompt=1280`, `max-num-seq=256`, `max-seq-len=2048`
| vLLM version | Chuncked Prefill | Scheduler Steps | Output Throughput (tokens/sec) | Median TPOT (ms) | Median ITL (ms) |
|--------------|------------------|-----------------|--------------------------------|------------------|-----------------|
| v0.5.4       | True             | N/A             | 1220.26           | 182.72      | 144.06     | 
| v0.6.0       | False            | 1               | 1331.25           | 163.52      | 114.42     | 
| v0.6.0       | False            | 10              | 1714.07           | 112.20      | 1036.33    | 


## Conclusion
The benchmarks demonstrate that NVIDIA H100 GPUs significantly outperform A100 GPUs, especially when handling larger models like the Llama and Mistral families. By leveraging tensor parallelism and optimizing batch sizes, the vLLM framework effectively balances throughput and latency, making it a powerful tool for large-scale LLM inference. For those aiming to optimize performance in similar contexts, utilizing H100 GPUs and adjusting parallelism settings may be particularly effective.

You can reproduce the benchmark with [this guide](https://github.com/LambdaLabsML/vllm-benchmark/blob/main/README.md). For users interested in self-hosting Llama 3.1 with vLLM, follow [this tutorial](https://docs.lambdalabs.com/on-demand-cloud/how-to-serve-the-llama-3.1-8b-and-70b-models-using-lambda-cloud-on-demand-instances?utm_source=linkedin&utm_medium=organic-social&utm_campaign=2024-09-vLLM-Benchmark-Report&utm_content=post-a) to deploy the model on [Lambda's Public Cloud](https://lambdalabs.com/service/gpu-cloud).

## Acknowledgement
We thank the [vLLM team](https://github.com/vllm-project/vllm) for providing their insights during this study. 
