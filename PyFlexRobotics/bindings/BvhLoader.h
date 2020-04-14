#ifndef BVH_LOADER_H
#define BVH_LOADER_H

#include <vector>
#include <string>

#include "../core/maths.h"
#include "../core/core.h"
// -------------------------------------------------
class BvhLoader
{
public:
    BvhLoader();
    ~BvhLoader();

    bool load(const std::string &filename);

    struct Bone
    {
        enum Channel
        {
            NO_CHANNEL,
            POS_X,
            POS_Y,
            POS_Z,
            ROT_X,
            ROT_Y,
            ROT_Z,
        };

        void init()
        {
            name = "";
            offset = Vec3(0.0f, 0.0f, 0.0f);
            parentNr = -1;
            for (int i = 0; i < 6; i++)
            {
                targetChannel[i] = NO_CHANNEL;
            }
            isEnd = false;
            endOffset = Vec3(0.0f, 0.0f, 0.0f);
        }
        std::string name;
        Vec3 offset;
        Channel targetChannel[6];
        int parentNr;

        bool isEnd;
        Vec3 endOffset;
    };

    void clear()
    {
        bones.clear();
        frames.clear();
        numFrames = 0;
        frameTime = 0.0f;
    };

    std::vector<Bone> bones;
    std::vector<float> frames;
    int numFrames;
    float frameTime;
};

#endif

