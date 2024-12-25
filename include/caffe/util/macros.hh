#pragma once

#define DISALLOW_COPY_AND_ASSIGN(classname) \
  classname(const classname&);              \
  classname& operator=(const classname&)

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>;   \
  template class classname<double>

#define IGNORE_VALUE(expr) (void)(expr)
