/* Copyright 2016 Jiang Chen <criver@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <queue>
#include <unistd.h>
#include <vector>

class ThreadPool {
public:
  ThreadPool(int num_threads);
  ~ThreadPool();

  void Enqueue(std::function<void()> f);

private:
  // Function that will be invoked by our threads.
  void Invoke();
  void ShutDown();

  std::vector<std::thread> thread_pool_;

  // Queue to keep track of incoming tasks.
  std::queue<std::function<void()>> tasks_;

  std::mutex queue_mutex_;
  std::condition_variable condition_;

  // Indicates that pool needs to be shut down.
  bool to_be_shutdown_ = false;
};
