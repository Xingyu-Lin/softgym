#ifndef ISAAC_UTIL_H_
#define ISAAC_UTIL_H_

#include <vector>
#include <string>
#include <chrono>

namespace IsaacIPC
{
    struct TimeVal
    {
        int hours           = 0;
        int minutes         = 0;
        int seconds         = 0;
        int milliseconds    = 0;
    };

    class Timer
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;

    public:

        Timer()
            : mStartTime(std::chrono::high_resolution_clock::now())
        {
        }

        void Restart()
        {
            mStartTime = std::chrono::high_resolution_clock::now();
        }

        double GetTime() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = now - mStartTime;
            return diff.count();
        }

        long GetMicroseconds() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            return (long)std::chrono::duration_cast<std::chrono::microseconds>(now - mStartTime).count();
        }

        long GetMilliseconds() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            return (long)std::chrono::duration_cast<std::chrono::milliseconds>(now - mStartTime).count();
        }

        TimeVal GetTimeVal() const
        {
            long t = GetMilliseconds();

            TimeVal tv;
            tv.milliseconds = t % 1000;
            t /= 1000;
            tv.seconds = t % 60;
            t /= 60;
            tv.minutes = t % 60;
            t /= 60;
            tv.hours = t;

            return tv;
        }
    };

    std::vector<char*>          TokenizeInPlace(char* str);
    inline std::vector<char*>   TokenizeInPlace(std::string& str)   { return TokenizeInPlace(&str[0]); }   // eek?
}

#endif
