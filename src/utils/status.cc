#include "status.h"

#include <string>

namespace gbdt {

Status::Status(error::ErrorCode code, const string& msg) : code_(code), msg_(msg) {
}

Status Status::OK() {
  return Status();
}

bool Status::ok() const {
  return code_ == error::OK;
}

const string& Status::error_msg() const {
  return msg_;
}

}  // namespace gbdt
