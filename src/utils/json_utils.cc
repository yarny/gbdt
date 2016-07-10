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

#include "json_utils.h"

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/util/type_resolver_util.h>
#include <string>

#include "src/base/base.h"

using google::protobuf::DescriptorPool;
using google::protobuf::Descriptor;
using google::protobuf::Message;
using google::protobuf::util::JsonOptions;
using google::protobuf::util::NewTypeResolverForDescriptorPool;
using google::protobuf::util::Status;
using google::protobuf::util::TypeResolver;

namespace {

static const char kTypeUrlPrefix[] = "type.googleapis.com";

static string GetTypeUrl(const Descriptor* message) {
  return string(kTypeUrlPrefix) + "/" + message->full_name();
}

}   // namespace

unique_ptr<TypeResolver> JsonUtils::resolver_(NewTypeResolverForDescriptorPool(
    kTypeUrlPrefix, DescriptorPool::generated_pool()));

JsonUtils::JsonUtils() {
}

Status JsonUtils::ToJson(const Message& message, string* str) {
  google::protobuf::util::JsonOptions options;
  options.always_print_primitive_fields = true;

  return BinaryToJsonString(resolver_.get(),
                            GetTypeUrl(message.GetDescriptor()),
                            message.SerializeAsString(), str, options);
}

Status JsonUtils::FromJson(const string& json, Message* message) {
  string binary;
  auto status = JsonToBinaryString(
      resolver_.get(), GetTypeUrl(message->GetDescriptor()), json, &binary);
  if (!status.ok()) return status;
  if (!message->ParseFromString(binary)) {
    return Status(error::INTERNAL, "Failed to parse protobuf.");
  }
  return Status::OK;
}
