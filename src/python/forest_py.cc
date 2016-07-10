#include "forest_py.h"

#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"

using gbdt::Forest;

class ForestPy {
 public:
  ForestPy(const string& str);
  ForestPy(Forest&& forest) : forest_(forest) {}

  string ToJson() const;

 private:
  Forest forest_;
};

ForestPy::ForestPy(const string& str) {
  Forest forest;
  if (!JsonUtils::FromJson(str, &forest)) {
    ThrowException(Status(gbdt::error::INVALID_ARGUMENT,
                          "Failed to parse Json."));

  }
  forest_ = std::move(forest);
}

string ForestPy::ToJson() const {
  return JsonUtils::ToJsonOrDie(forest_);
}

void InitForestPy(py::module &m) {
  py::class_<ForestPy>(m, "Forest")
      .def(py::init<const string&>())
      .def("as_json", &ForestPy::ToJson);
}
