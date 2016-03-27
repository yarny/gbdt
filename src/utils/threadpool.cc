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

#include "threadpool.h"

using namespace::std;

ThreadPool::ThreadPool(int num_threads) {
  // Create number of required threads and add them to the thread pool vector.
  for(int i = 0; i < num_threads; ++i) {
    thread_pool_.emplace_back(thread(&ThreadPool::Invoke, this));
  }
}

void ThreadPool::Enqueue(std::function<void()> f) {
  {
    unique_lock<mutex> lock(queue_mutex_);
    tasks_.push(f);
  }

  // Wake up one thread.
  condition_.notify_one();
}

void ThreadPool::Invoke() {
  while(true) {
    function<void()> task;

    {
      // Put unique lock on task mutex.
      unique_lock<mutex> lock(queue_mutex_);

      // Wait until queue is not empty or termination signal is sent.
      condition_.wait(lock, [this]{ return !tasks_.empty() || to_be_shutdown_; });

      // If termination signal received and queue is empty then exit else continue
      // clearing the queue.
      if (to_be_shutdown_ && tasks_.empty()) {
        return;
      }

      // Get next task in the queue.
      task = tasks_.front();
      tasks_.pop();
    }

    // Execute the task.
    task();
  }
}

void ThreadPool::ShutDown() {
  {
    // Put unique lock on task mutex.
    unique_lock<mutex> lock(queue_mutex_);
    to_be_shutdown_ = true;
  }

  // Wake up all threads.
  condition_.notify_all();

  // Join all threads.
  for(auto& thread : thread_pool_) {
    thread.join();
  }
}

ThreadPool::~ThreadPool() {
  ShutDown();
}
