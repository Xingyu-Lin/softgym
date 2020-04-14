#include "IsaacUtil.h"

#include <cctype>

// local helpers
namespace
{
    char* NextToken(char*& p)
    {
        if (!p) {
            return nullptr;
        }

        // skip leading whitespace
        while (isspace(*p)) {
            ++p;
        }

        // blank input?
        if (!*p) {
            return nullptr;
        }

        char* tokStart = p;

        // skip token
        while (*p && !isspace(*p)) {
            ++p;
        }

        // nul-terminate the token (!!! intrusive !!!)
        if (*p) {
            *p++ = '\0';
        }

        return tokStart;
    }
}

namespace IsaacIPC
{
    std::vector<char*> TokenizeInPlace(char* str)
    {
        std::vector<char*> tokens;
        for (char* tok = NextToken(str); tok; tok = NextToken(str)) {
            tokens.push_back(tok);
        }
        return tokens;
    }
}
