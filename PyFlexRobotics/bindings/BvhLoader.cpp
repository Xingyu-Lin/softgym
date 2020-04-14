#include "BvhLoader.h"

#include <string.h>

#if !_WIN32
#define strnicmp strncasecmp
#endif

#define LINE_LEN 10000

// ------------------------------------------------------------------------------------
BvhLoader::BvhLoader()
{
    clear();
}

// ------------------------------------------------------------------------------------
BvhLoader::~BvhLoader()
{

}

// ------------------------------------------------------------------------------------
bool BvhLoader::load(const std::string &filename)
{
    clear();



    char line[LINE_LEN];
    char name[1024];
    char val[1024];

    std::vector<int> boneStack;

    bool readingMotion = false;

    FILE *f = fopen(filename.c_str(), "r");
    if (f == NULL)
    {
        return false;
    }

    while (!feof(f))
    {
        if (!fgets(line, LINE_LEN, f))
        {
            break;
        }

        char *s = line;
        while (*s != '\0' && *s <= ' ')
        {
            s++;
        }
        if (*s == '\0')
        {
            continue;
        }

        if (strnicmp(s, "HIERARCHY", 9) == 0)
        {
            continue;
        }
        else if (strnicmp(s, "ROOT", 4) == 0)
        {
            sscanf(s + 5, "%s", name);
            Bone b;
            b.init();
            b.name = name;
            bones.push_back(b);
            boneStack.push_back(0);
        }
        else if (strnicmp(s, "OFFSET", 6) == 0 && !boneStack.empty())
        {
            Bone &b = bones[boneStack.back()];
            Vec3 &offset = b.isEnd ? b.endOffset : b.offset;
            sscanf(s + 7, "%f %f %f", &offset.x, &offset.y, &offset.z);
        }
        else if (strnicmp(s, "CHANNELS", 8) == 0 && !boneStack.empty())
        {
            Bone &b = bones[boneStack.back()];
            int numChannels = -1;
            s += 8;
            int nr = 0;
            while (*s != '\0')
            {
                while (*s != '\0' && *s <= ' ')
                {
                    s++;
                }
                int i = 0;
                while (*s != '\0' && *s > ' ')
                {
                    val[i++] = *s++;
                }
                val[i] = '\0';
                if (numChannels < 0)
                {
                    sscanf(val, "%i", &numChannels);
                }
                else if (strnicmp(val, "Xposition", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::POS_X;
                }
                else if (strnicmp(val, "Yposition", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::POS_Y;
                }
                else if (strnicmp(val, "Zposition", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::POS_Z;
                }
                else if (strnicmp(val, "Xrotation", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::ROT_X;
                }
                else if (strnicmp(val, "Yrotation", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::ROT_Y;
                }
                else if (strnicmp(val, "Zrotation", 9) == 0)
                {
                    b.targetChannel[nr++] = Bone::ROT_Z;
                }
            }
        }
        else if (strnicmp(s, "JOINT", 5) == 0 && !boneStack.empty())
        {
            sscanf(s + 6, "%s", name);
            Bone b;
            b.init();
            b.name = name;
            b.parentNr = boneStack.back();
            boneStack.push_back(int(bones.size()));
            bones.push_back(b);
        }
        else if (strnicmp(s, "End Site", 8) == 0 && !boneStack.empty())
        {
            bones[boneStack.back()].isEnd = true;
            boneStack.push_back(boneStack.back());
        }
        else if (*s == '}')
        {
            boneStack.pop_back();
        }
        else if (strnicmp(s, "MOTION", 6) == 0)
        {
            readingMotion = true;
        }
        else if (strnicmp(s, "FRAMES", 6) == 0)
        {
            s += 6;
            while (*s != '\0' && (*s < '0' || *s > '9'))
            {
                s++;
            }
            sscanf(s, "%i", &numFrames);
        }
        else if (strnicmp(s, "FRAME TIME", 10) == 0)
        {
            s += 10;
            while (*s != '\0' && (*s < '.' || *s > '9'))
            {
                s++;
            }
            sscanf(s, "%f", &frameTime);
        }
        else if (readingMotion)
        {
            while (*s != '\0')
            {
                while (*s != '\0' && *s < '-')
                {
                    s++;
                }
                if (*s == '\0')
                {
                    break;
                }

                strcpy(val, "");
                int i = 0;
                while (*s != '\0' && *s >= '-')
                {
                    val[i++] = *s++;
                }
                val[i] = '\0';
                float valf = 0.0f;
                sscanf(val, "%f", &valf);
                frames.push_back(valf);
            }
        }
    }

    fclose(f);
    return true;
}