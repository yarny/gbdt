#include "gbdt_py_base.h"

#include <ios>
#include <stdexcept>

void ThrowException(const gbdt::Status& status) {
  switch(status.error_code()) {
    case gbdt::error::INVALID_ARGUMENT:
      throw std::invalid_argument(status.error_msg());
    case gbdt::error::OUT_OF_RANGE:
      throw std::out_of_range(status.error_msg());
    case gbdt::error::NOT_FOUND:
      throw std::ios::failure(status.error_msg());
    case gbdt::error::INVALID_OPERATION:
      throw std::logic_error(status.error_msg());
    case gbdt::error::OK:
      return;
  }
}
