#include <gtest/gtest.h>

// make array pointer comparable for testing by converting it into a tuple
MATCHER_P2(ArrayPointee, size, subMatcher, "")
{
    return ExplainMatchResult(subMatcher, std::make_tuple(arg, size), result_listener);
}