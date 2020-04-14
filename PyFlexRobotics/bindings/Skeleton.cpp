#include "Skeleton.h"
#include "BvhLoader.h"

#include <string.h>

#if !_WIN32
#define strnicmp strncasecmp
#endif


// ------------------------------------------------------------------------------------
Skeleton::Skeleton()
{
    mAnimNr = 0;
}

// ------------------------------------------------------------------------------------
Skeleton::~Skeleton()
{

}

// ------------------------------------------------------------------------------------
void Skeleton::clear()
{
    mBones.clear();
    mAnimations.clear();
    mAnimNr = 0;
    mCurrentRootPos = Vec3(0.0f, 0.0f, 0.0f);
};


// ------------------------------------------------------------------------------------
void Skeleton::finalize()
{
    evaluateGlobalBindPose();

    mLower = Vec3(1e7f, 1e7f, 1e7f);
    mUpper = Vec3(-1e7f, -1e7f, -1e7f);
    setBindPose();

    mBoneTransforms.resize(mBones.size(), Transform());
    for (int i = 0; i < (int)mBones.size(); i++)
    {

        Vec3& p = mBones[i].globalBindPose.p;
        if (p.x < mLower.x)
        {
            mLower.x = p.x;
        }
        if (p.y < mLower.y)
        {
            mLower.y = p.y;
        }
        if (p.z < mLower.z)
        {
            mLower.z = p.z;
        }

        if (p.x > mUpper.x)
        {
            mUpper.x = p.x;
        }
        if (p.y > mUpper.y)
        {
            mUpper.y = p.y;
        }
        if (p.z > mUpper.z)
        {
            mUpper.z = p.z;
        }
    }
}
Quat getConjugate(Quat q)
{
    return Quat(-q.x, -q.y, -q.z, q.w);
}
// ------------------------------------------------------------------------------------
float readNext(char* &s)
{
    while (*s != 0 && *s <= ' ')
    {
        s++;
    }
    if (*s == 0)
    {
        return 0.0f;
    }

    char val[256];
    int i = 0;

    while (*s != 0 && *s > ' ')
    {
        val[i++] = *s++;
    }
    val[i] = 0;

    float f;
    sscanf(val, "%f", &f);
    return f;
}


// ------------------------------------------------------------------------------------
bool Skeleton::load(const std::string &filename)
{
    FILE *f = fopen(filename.c_str(), "r");
    if (f != 0)
    {
        return false;
    }

    clear();

    char s[10000], name[256];

    while (!feof(f))
    {
        if (fgets(s, 10000, f) == NULL)
        {
            break;
        }

        if (strnicmp(s, "bone", 4) == 0)
        {
            mBones.resize(mBones.size() + 1);
            mBones.back().init();
            sscanf(s + 5, "%s", name);
            mBones.back().name = name;
        }
        else if (strnicmp(s, "parent", 6) == 0)
        {
            sscanf(s + 7, "%i", &mBones.back().parentNr);
        }
        else if (strnicmp(s, "localParentOffset", 17) == 0)
        {
            Vec3 &off = mBones.back().localParentOffset;
            sscanf(s + 18, "%f %f %f", &off.x, &off.y, &off.z);
        }
        else if (strnicmp(s, "localChildOffset", 16) == 0)
        {
            Vec3 &off = mBones.back().localChildOffset;
            sscanf(s + 17, "%f %f %f", &off.x, &off.y, &off.z);
        }
        else if (strnicmp(s, "localBindRot", 12) == 0)
        {
            Quat &rot = mBones.back().bindPose;
            sscanf(s + 13, "%f %f %f %f", &rot.x, &rot.y, &rot.z, &rot.w);
        }


        else if (strnicmp(s, "animation", 9) == 0)
        {
            mAnimations.resize(mAnimations.size() + 1);
            mAnimations.back().init();
            sscanf(s + 10, "%s", name);
            mAnimations.back().name = name;
        }
        else if (strnicmp(s, "frameTime", 9) == 0)
        {
            sscanf(s + 10, "%f", &mAnimations.back().frameTime);
        }
        else if (strnicmp(s, "numFrames", 9) == 0)
        {
            //Vec3 &off = mBones.back().localParentOffset;
            sscanf(s + 10, "%i", &mAnimations.back().numFrames);
        }
        else if (strnicmp(s, "pose", 4) == 0)
        {
            char *p = s + 5;
            Animation &a = mAnimations.back();
            Vec3 pos;
            pos.x = readNext(p);
            pos.y = readNext(p);
            pos.z = readNext(p);
            a.rootPos.push_back(pos);
            for (int i = 0; i < (int)mBones.size(); i++)
            {
                Quat q;
                q.x = readNext(p);
                q.y = readNext(p);
                q.z = readNext(p);
                q.w = readNext(p);
                a.jointRotations.push_back(q);
            }
        }
    }
    fclose(f);

    finalize();

    return true;
}

// ------------------------------------------------------------------------------------
bool Skeleton::save(const std::string &filename) const
{
    if (mBones.empty())
    {
        return false;
    }

    FILE *f = fopen(filename.c_str(), "w");
    if (f != 0)
    {
        return false;
    }

    int numBones = int(mBones.size());

    fprintf(f, "skeleton\n");
    for (int i = 0; i < numBones; i++)
    {
        const Bone &b = mBones[i];
        fprintf(f, "\nbone %s\n", b.name.c_str());
        fprintf(f, "parent %i\n", b.parentNr);
        fprintf(f, "localParentOffset %f %f %f\n", b.localParentOffset.x, b.localParentOffset.y, b.localParentOffset.z);
        fprintf(f, "localChildOffset %f %f %f\n", b.localChildOffset.x, b.localChildOffset.y, b.localChildOffset.z);
        fprintf(f, "localBindRot %f %f %f %f\n", b.bindPose.x, b.bindPose.y, b.bindPose.z, b.bindPose.w);
    }

    for (int i = 0; i < (int)mAnimations.size(); i++)
    {
        const Animation &a = mAnimations[i];
        fprintf(f, "\nanimation %s\n", a.name.c_str());
        fprintf(f, "frameTime %f\n", a.frameTime);
        fprintf(f, "numFrames %i\n", a.numFrames);
        fprintf(f, "// pose rootpos.x rootpos.y rootpos.z bone0.localquat.x bone0.localquat.y bone0.localquat.z bone0.localquat.w bone1...\n");
        for (int j = 0; j < a.numFrames; j++)
        {
            fprintf(f, "pose %f %f %f  ", a.rootPos[j].x, a.rootPos[j].y, a.rootPos[j].z);
            for (int k = 0; k < numBones; k++)
            {
                const Quat &q = a.jointRotations[j * numBones + k];
                fprintf(f, "%f %f %f %f ", q.x, q.y, q.z, q.w);
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
    return true;

}

// ------------------------------------------------------------------------------------
bool Skeleton::loadFromBvh(const std::string &filename)
{
    float scale = 0.01f;

    BvhLoader loader;
    if (!loader.load(filename))
    {
        return false;
    }

    if (loader.bones.empty() || loader.frames.empty())
    {
        return false;
    }

    std::vector<int> numChildren(loader.bones.size(), 0);
    std::vector<Matrix33> frames(loader.bones.size());

    clear();
    mBones.resize(loader.bones.size());
    for (int i = 0; i < (int)loader.bones.size(); i++)
    {
        BvhLoader::Bone &lb = loader.bones[i];

        Bone &b = mBones[i];
        b.init();
        b.name = lb.name;
        b.parentNr = lb.parentNr;
        b.localParentOffset = lb.offset * scale;
        if (b.parentNr >= 0)
        {
            mBones[b.parentNr].localChildOffset += lb.offset * scale;
            numChildren[b.parentNr]++;
        }
        if (lb.isEnd)
        {
            b.localChildOffset = lb.endOffset * scale;
        }
    }

    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];
        if (numChildren[i] > 0)
        {
            b.localChildOffset /= (float)numChildren[i];
        }

        Matrix33 parentFrame = b.parentNr >= 0 ? frames[b.parentNr] : Matrix33::Identity();
        Matrix33 &frame = frames[i];

        frame.cols[0] = b.localChildOffset;
        if (Dot(frame.cols[0], frame.cols[0]) < 1e-5f)
        {
            frame.cols[0] = Vec3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            frame.cols[0] = Normalize(frame.cols[0]);
        }
        // transport previous frame
        if (fabs(Dot(parentFrame.cols[1], frame.cols[0])) <
                fabs(Dot(parentFrame.cols[2], frame.cols[0])))
        {
            frame.cols[1] = parentFrame.cols[1];
            frame.cols[1] -= frame.cols[0] * Dot(frame.cols[0], frame.cols[1]);
            frame.cols[1] = Normalize(frame.cols[1]);
            frame.cols[2] = Cross(frame.cols[0], frame.cols[1]);
        }
        else
        {
            frame.cols[2] = parentFrame.cols[2];
            frame.cols[2] -= frame.cols[0] * Dot(frame.cols[0], frame.cols[2]);
            frame.cols[2] = Normalize(frame.cols[2]);
            frame.cols[1] = Cross(frame.cols[2], frame.cols[0]);
        }

        if (b.parentNr >= 0)
        {
            b.localParentOffset = Transpose(parentFrame)*b.localParentOffset;
        }
        b.localChildOffset = Transpose(frames[i])*b.localChildOffset;

        Quat q = Quat(frames[i]);
        Quat p = Quat(parentFrame);
        b.bindPose = getConjugate(p) * q;
    }

    // animation

    mAnimations.resize(mAnimations.size() + 1);
    Animation &a = mAnimations.back();
    a.numFrames = loader.numFrames;
    a.frameTime = loader.frameTime;

    int i = int(filename.find_last_of("/\\:") + 1);
    int j = int(filename.find_last_of("."));
    if (j == int(std::string::npos))
    {
        j = int(filename.size());
    }

    a.name = filename.substr(i, j - i);

    float *f = &loader.frames[0];
    for (int i = 0; i < loader.numFrames; i++)
    {
        for (int j = 0; j < (int)loader.bones.size(); j++)
        {
            BvhLoader::Bone &lb = loader.bones[j];
            Vec3 pos = Vec3(0.0f, 0.0f, 0.0f);
            Quat rot = Quat();
            float s = kPi / 180.0f;

            for (int k = 0; k < 6; k++)
            {
                switch (lb.targetChannel[k])
                {
                case BvhLoader::Bone::POS_X:
                    pos.x = *f++;
                    break;
                case BvhLoader::Bone::POS_Y:
                    pos.y = *f++;
                    break;
                case BvhLoader::Bone::POS_Z:
                    pos.z = *f++;
                    break;
                case BvhLoader::Bone::ROT_X:
                    rot = rot * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.f), *f++ * s);
                    break;
                case BvhLoader::Bone::ROT_Y:
                    rot = rot * QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.f), *f++ * s);
                    break;
                case BvhLoader::Bone::ROT_Z:
                    rot = rot * QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.f), *f++ * s);
                    break;
				case BvhLoader::Bone::NO_CHANNEL:
					break;
                }
            }

            Quat prevQ = lb.parentNr >= 0 ? Quat(frames[lb.parentNr]) : Quat();
            Quat bindQ = mBones[j].bindPose;

            a.jointRotations.push_back(getConjugate(prevQ) * rot * prevQ * bindQ);
            if (lb.parentNr < 0)
            {
                a.rootPos.push_back(pos * scale);
            }
        }
    }

    mAnimNr = 0;

    finalize();

    return true;
}

// ------------------------------------------------------------------------------------
bool Skeleton::addAnimationFromBvh(const std::string &filename)
{
    Skeleton *s = new Skeleton();
    if (!s->loadFromBvh(filename))
    {
        delete s;
        return false;
    }

    std::vector<int> oldToNew(s->mBones.size(), -1);
    for (int i = 0; i < (int)s->mBones.size(); i++)
    {
        for (int j = 0; j < (int)mBones.size(); j++)
        {
            if (s->mBones[i].name.compare(mBones[j].name) == 0)
            {
                oldToNew[i] = j;
            }
        }
    }

    for (int i = 0; i < (int)s->mAnimations.size(); i++)
    {
        Animation &in = s->mAnimations[i];
        mAnimations.resize(mAnimations.size() + 1);
        Animation &out = mAnimations.back();
        out.init();
        out.jointRotations.resize(in.numFrames * mBones.size(), Quat());
        out.numFrames = in.numFrames;
        out.frameTime = in.frameTime;
        out.name = in.name;
        out.rootPos = in.rootPos;

        for (int j = 0; j < in.numFrames; j++)
        {
            Quat *inRots = &in.jointRotations[j * s->mBones.size()];
            Quat *outRots = &out.jointRotations[j * mBones.size()];

            for (int k = 0; k < (int)s->mBones.size(); k++)
            {
                if (oldToNew[k] < 0)
                {
                    continue;
                }

                Quat q = inRots[k];
                int nr = s->mBones[k].parentNr;
                while (nr >= 0 && oldToNew[nr] < 0)
                {
                    q = inRots[nr] * q;
                    nr = s->mBones[nr].parentNr;
                }
                outRots[oldToNew[k]] = q;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------------------------------
void Skeleton::setBindPose()
{
    int numBones = int(mBones.size());

    for (int i = 0; i < numBones; i++)
    {
        Bone &b = mBones[i];
        b.pose = b.bindPose;
    }
    evaluateGlobalPose();
}

// ------------------------------------------------------------------------------------
void Skeleton::setAnimNr(int nr)
{
    mAnimNr = nr;
    if (mAnimNr < 0)
    {
        mAnimNr = 0;
    }
    if (mAnimNr >= (int)mAnimations.size())
    {
        mAnimNr = int(mAnimations.size() - 1);
    }
}

// ------------------------------------------------------------------------------------
void Skeleton::setAnimFrame(int frameNr)
{
    if (mAnimations.empty())
    {
        return;
    }

    Animation &a = mAnimations[mAnimNr];
    if (frameNr < 0 || frameNr >= a.numFrames)
    {
        return;
    }

    int numBones = int(mBones.size());

    for (int i = 0; i < numBones; i++)
    {
        Bone &b = mBones[i];
        b.pose = a.jointRotations[frameNr * numBones + i];
    }
    mCurrentRootPos = a.rootPos[frameNr];
    evaluateGlobalPose();
}

// ------------------------------------------------------------------------------------
void Skeleton::setAnimTime(float time)
{
    if (mAnimNr < 0 || mAnimNr >= (int)mAnimations.size())
    {
        return;
    }

    Animation &a = mAnimations[mAnimNr];

    float length = a.numFrames * a.frameTime;
    if (time < 0.0f)
    {
        time = 0.0f;
    }
    if (time > length)
    {
        time = length;
    }

    int nr0 = (int)floorf(time / a.frameTime);
    if (nr0 >= a.numFrames - 1)
    {
        nr0 = a.numFrames - 2;
    }
    int nr1 = nr0 + 1;

    float s = (time - nr0 * a.frameTime) / a.frameTime;

    int numBones = int(mBones.size());

    for (int i = 0; i < numBones; i++)
    {
        Bone &b = mBones[i];

        Quat q0 = a.jointRotations[nr0 * numBones + i];
        Quat q1 = a.jointRotations[nr1 * numBones + i];

        Quat q = q0 * (1.0f - s) + q1 * s;
        q = Normalize(q);
        b.pose = q;
    }
    mCurrentRootPos = (1.0f - s) * a.rootPos[nr0] + s * a.rootPos[nr1];
    evaluateGlobalPose();
}

// ------------------------------------------------------------------------------------
void Skeleton::evaluateGlobalPose(int modBoneNr, const Quat &globalQ)
{
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];

        Transform parentPose = b.parentNr >= 0 ? mBones[b.parentNr].globalPose : Transform();
        if (b.parentNr < 0)
        {
            parentPose.p += mCurrentRootPos;
        }

        b.globalPose.p = parentPose.p;
        if (b.parentNr >= 0)
        {
            b.globalPose.p += Rotate(parentPose.q, b.localParentOffset);
        }
        b.globalPose.q = parentPose.q * b.pose;
        if (i == modBoneNr)
        {
            b.globalPose.q = globalQ * b.globalPose.q;
        }
    }

    mBoneTransforms.resize(mBones.size());
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];
        mBoneTransforms[i] = b.globalPose * Inverse(b.globalBindPose);
    }
}

// ------------------------------------------------------------------------------------
void Skeleton::evaluateGlobalBindPose(int modBoneNr, const Quat &globalQ)
{
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];

        Transform parentPose = b.parentNr >= 0 ? mBones[b.parentNr].globalBindPose : Transform();

        b.globalBindPose.p = parentPose.p + Rotate(parentPose.q,b.localParentOffset);
        b.globalBindPose.q = parentPose.q * b.bindPose;
        if (i == modBoneNr)
        {
            b.globalBindPose.q = globalQ * b.globalBindPose.q;
        }
    }
}

// ------------------------------------------------------------------------------------
void Skeleton::readBackPose()
{
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];

        Transform parentPose = b.parentNr >= 0 ? mBones[b.parentNr].globalPose : Transform();
        b.pose = getConjugate(parentPose.q) * b.globalPose.q;
        b.pose = Normalize(b.pose);
    }
}

// ------------------------------------------------------------------------------------
void Skeleton::readBackBindPose()
{
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        Bone &b = mBones[i];

        Transform parentPose = b.parentNr >= 0 ? mBones[b.parentNr].globalBindPose : Transform();
        b.bindPose = getConjugate(parentPose.q) * b.globalBindPose.q;
        b.bindPose = Normalize(b.bindPose);
    }
}

// ------------------------------------------------------------------------------------
bool Skeleton::deleteBone(int nr)
{
    int numBones = int(mBones.size());
    if (nr == 0 || nr >= numBones)
    {
        return false;
    }

    // animation
    for (int i = 0; i < (int)mAnimations.size(); i++)
    {
        Animation &a = mAnimations[i];
        int num = 0;
        for (int j = 0; j < a.numFrames; j++)
        {

            Quat *rots = &a.jointRotations[j * numBones];
            for (int k = 0; k < numBones; k++)
            {
                if (mBones[k].parentNr == nr)
                {
                    rots[k] = rots[nr] * rots[k];
                }
            }
            for (int k = 0; k < numBones; k++)
            {
                if (k != nr)
                {
                    a.jointRotations[num++] = rots[k];
                }
            }
        }
        a.jointRotations.resize(num);
    }

    // skeleton
    for (int i = 0; i < numBones; i++)
    {
        if (mBones[i].parentNr == nr)
        {
            mBones[i].parentNr = mBones[nr].parentNr;
            mBones[i].localParentOffset = mBones[nr].localParentOffset;
            mBones[i].bindPose = mBones[nr].bindPose * mBones[i].bindPose;
            mBones[i].pose = mBones[nr].pose * mBones[i].pose;
        }
        if (mBones[i].parentNr > nr)
        {
            mBones[i].parentNr--;
        }
    }

    for (int i = nr; i < numBones - 1; i++)
    {
        mBones[i] = mBones[i + 1];
    }
    mBones.pop_back();

    evaluateGlobalBindPose();
    evaluateGlobalPose();

    return true;
}

// ------------------------------------------------------------------------------------
void Skeleton::mirrorBindPose(bool leftToRight)
{
    for (int i = 0; i < (int)mBones.size(); i++)
    {
        for (int j = 0; j < (int)mBones.size(); j++)
        {
            if (i == j)
            {
                continue;
            }
            std::string &ni = mBones[i].name;
            std::string &nj = mBones[j].name;
            if (ni.size() != nj.size())
            {
                continue;
            }

            bool match = true;

            for (int k = 0; k < (int)ni.size(); k++)
            {
                bool charMatch = ni[k] == nj[k];
                if (leftToRight)
                {
                    charMatch = charMatch || (ni[k] == 'l' && nj[k] == 'r') || (ni[k] == 'L' && nj[k] == 'R');
                }
                else
                {
                    charMatch = charMatch || (ni[k] == 'r' && nj[k] == 'l') || (ni[k] == 'R' && nj[k] == 'L');
                }

                if (!charMatch)
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                Bone &bi = mBones[i];
                Bone &bj = mBones[j];
                Transform &t = bi.globalBindPose;
                bj.globalBindPose.p = Vec3(-t.p.x, t.p.y, t.p.z);
                bj.globalBindPose.q = Quat(-t.q.y, -t.q.x, t.q.w, t.q.z);

                Vec3 p = bi.localParentOffset;
                bj.localParentOffset = Vec3(p.x, -p.y, p.z);
                p = bi.localChildOffset;
                bj.localChildOffset = Vec3(p.x, -p.y, p.z);
            }
        }
    }
    readBackBindPose();
    evaluateGlobalBindPose();
}


// ------------------------------------------------------------------------------------
void Skeleton::readBackAnimFrame(float time)
{
    if (mAnimNr < 0 || mAnimNr >= (int)mAnimations.size())
    {
        return;
    }

    Animation &a = mAnimations[mAnimNr];
    int frameNr = int(time / (float)a.frameTime);
    if (frameNr < 0)
    {
        frameNr = 0;
    }
    if (frameNr >= a.numFrames)
    {
        frameNr = a.numFrames - 1;
    }

    a.rootPos[frameNr] = mCurrentRootPos;
    int numBones = int(mBones.size());

    for (int i = 0; i < numBones; i++)
    {
        a.jointRotations[frameNr * numBones + i] = mBones[i].pose;
    }
}

float Skeleton::getAnimLength()
{
    Animation &a = mAnimations[mAnimNr];
    float length = a.numFrames * a.frameTime;
    return length;
}

