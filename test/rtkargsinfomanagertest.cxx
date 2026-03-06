#include <cstdlib>
#include <iostream>
#include <rtkGgoArgsInfoManager.h>

/**
 * \file rtkargsinfomanagertest.cxx
 *
 * \brief Tests the cleanup behavior of `rtk::args_info_manager`.
 *
 * This test checks that the `args_info_manager` calls the provided cleanup
 * function or functor on destruction, so the wrapped `args_info_test` is
 * updated accordingly.
 *
 * \author Lucas Gandel
 */

class args_info_test
{
public:
  int testVar{ true };
  int verbose_flag{ 0 };
  args_info_test() = default;
};

class cleanup_functor
{
public:
  void
  operator()(args_info_test * args_info)
  {
    args_info->testVar = false;
  }
};

void
cleanup_function(args_info_test * args_info)
{
  args_info->testVar = false;
}

int
main(int, char **)
{
  args_info_test args_info_1, args_info_2;

  { // new scope - manager does cleanup on destruction
    rtk::args_info_manager<args_info_test> manager_1(args_info_1, cleanup_function);

    cleanup_functor                                         functor;
    rtk::args_info_manager<args_info_test, cleanup_functor> manager_2(args_info_2, functor);
  }

  if (args_info_1.testVar)
  {
    std::cout << "Test FAILED -- cleanup using a function didn't work." << std::endl;
    return EXIT_FAILURE;
  }

  if (args_info_2.testVar)
  {
    std::cout << "Test FAILED -- cleanup using a functor didn't work." << std::endl;
    return EXIT_FAILURE;
  }

  // otherwise
  return EXIT_SUCCESS;
}
