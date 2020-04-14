#ifndef SKELETON_H
#define SKELETON_H

#include <vector>
#include <string>

#include "../core/maths.h"
#include "../core/core.h"

// -------------------------------------------------
class Skeleton
{
public:
    Skeleton();
    ~Skeleton();

    void clear();

    bool loadFromBvh(const std::string &filename);
    bool addAnimationFromBvh(const std::string &filename);

    bool load(const std::string &filename);
    bool save(const std::string &filename) const;

    struct Bone
    {
        void init()
        {
            name = "";
            parentNr = -1;
            localParentOffset = Vec3(0.0f, 0.0f, 0.0f);
            localChildOffset = Vec3(0.0f, 0.0f, 0.0f);
            bindPose = Quat();
            globalBindPose = Transform();
            pose = Quat();
            globalPose = Transform();
        }
        std::string name;
        int parentNr;
        Vec3 localParentOffset;
        Vec3 localChildOffset;
        Quat bindPose;
        Transform globalBindPose;
        Quat pose;
        Transform globalPose;
    };

    struct Animation
    {
        void init()
        {
            name = "";
            frameTime = 0.0f;
            numFrames = 0;
            jointRotations.clear();
            rootPos.clear();
        }
        std::string name;
        float frameTime;
        int numFrames;
        std::vector<Quat> jointRotations;
        std::vector<Vec3> rootPos;
    };

    std::vector<Bone> &getBones()
    {
        return mBones;
    }
    std::vector<Animation> &getAnimations()
    {
        return mAnimations;
    }
    Animation &getCurrentAnimation()
    {
        return mAnimations[mAnimNr];
    }

    void setBindPose();

    void setAnimNr(int nr);
    int getAnimNr() const
    {
        return mAnimNr;
    }

    void setAnimFrame( int frameNr);
    void setAnimTime(float time);
    void evaluateGlobalPose(int modBoneNr = -1, const Quat &globalQ = Quat());
    void evaluateGlobalBindPose(int modBoneNr = -1, const Quat &globalQ = Quat());
    void readBackPose();
    void readBackBindPose();

    void readBackAnimFrame(float time);

    void setCurrentRootPos(const Vec3 &pos)
    {
        mCurrentRootPos = pos;
    }
    const Vec3 &getCurrentRootPos() const
    {
        return mCurrentRootPos;
    }

    const Vec3 &getLower()
    {
        return mLower;
    }
    const Vec3 &getUpper()
    {
        return mUpper;
    }

    bool deleteBone(int nr);
    void mirrorBindPose(bool leftToRight);

    std::vector<Transform> &getBoneTransforms()
    {
        return mBoneTransforms;
    }

    float getAnimLength();

private:
    void finalize();

    std::vector<Bone> mBones;
    std::vector<Transform> mBoneTransforms;

    std::vector<Animation> mAnimations;
    int mAnimNr;
    Vec3 mCurrentRootPos;
    Vec3 mLower;
    Vec3 mUpper;
};

#endif

