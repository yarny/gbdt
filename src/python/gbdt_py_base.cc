#include "gbdt_py_base.h"

#include <ios>
#include <stdexcept>

void ThrowException(const Status& status) {
  switch(status.error_code()) {
    case error::INVALID_ARGUMENT:
      throw std::invalid_argument(status.ToString());
    case error::OUT_OF_RANGE:
      throw std::out_of_range(status.ToString());
    case error::NOT_FOUND:
      throw std::ios::failure(status.ToString());
    case error::FAILED_PRECONDITION:
      throw std::logic_error(status.ToString());
    case error::INTERNAL:
      throw std::logic_error(status.ToString());
    case error::OK:
      return;
  }
}
