#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <iterator>

class SoftgymTshirt : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;
    char tshirt_path[100];

    SoftgymTshirt(const char* name) : Scene(name) {}

    char* make_path(char* full_path, std::string path) {
        strcpy(full_path, getenv("PYFLEXROOT"));
        strcat(full_path, path.c_str());
        return full_path;
    }

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    void sortInd(uint32_t* a, uint32_t* b, uint32_t* c)
    {
        if (*b < *a)
            swap(a,b);

        if (*c < *b)
        {
            swap(b,c);
            if (*b < *a)
                swap(b, a);
        }
    }


    void findUnique(map<uint32_t, uint32_t> &unique, Mesh* m)
    {
        map<vector<float>, uint32_t> vertex;
        map<vector<float>, uint32_t>::iterator it;

        uint32_t count = 0;
        for (uint32_t i=0; i < m->GetNumVertices(); ++i)
        {
            Point3& v = m->m_positions[i];
            float arr[] = {v.x, v.y, v.z};
            vector<float> p(arr, arr + sizeof(arr)/sizeof(arr[0]));

            it = vertex.find(p);
            if (it == vertex.end()) {
                vertex[p] = i;
                unique[i] = i;
                count++;
            }
            else
            {
                unique[i] = it->second;
            }
        }

        cout << "total vert:  " << m->GetNumVertices() << endl;
        cout << "unique vert: " << count << endl;
    }


    void createTshirt(const char* filename, Vec3 lower, float scale, float rotation, Vec3 velocity, int phase, float invMass, float stiffness)
    {
        // import the mesh
        Mesh* m = ImportMesh(filename);
        if (!m)
            return;

        // rotate mesh
        m->Transform(RotationMatrix(rotation, Vec3(0.0f, 1.0f, 0.0f)));

        Vec3 meshLower, meshUpper;
        m->GetBounds(meshLower, meshUpper);

        Vec3 edges = meshUpper-meshLower;
        float maxEdge = max(max(edges.x, edges.y), edges.z);

        // put mesh at the origin and scale to specified size
        Matrix44 xform = ScaleMatrix(scale/maxEdge)*TranslationMatrix(Point3(-meshLower));

        m->Transform(xform);
        m->GetBounds(meshLower, meshUpper);

        // index of particles
        uint32_t baseIndex = uint32_t(g_buffers->positions.size());
        uint32_t currentIndex = baseIndex;

        // find unique vertices
        map<vector<float>, uint32_t> vertex;
        map<vector<float>, uint32_t>::iterator it;
        
        // maps from vertex index to particle index
        map<uint32_t, uint32_t> indMap;

        // to check for duplicate connections
        map<uint32_t,list<uint32_t> > edgeMap;

        // loop through the faces
        for (uint32_t i=0; i < m->GetNumFaces(); ++i)
        {
            // create particles
            uint32_t a = m->m_indices[i*3+0];
            uint32_t b = m->m_indices[i*3+1];
            uint32_t c = m->m_indices[i*3+2];

            Point3& v0 = m->m_positions[a];
            Point3& v1 = m->m_positions[b];
            Point3& v2 = m->m_positions[c];

            float arr0[] = {v0.x, v0.y, v0.z};
            float arr1[] = {v1.x, v1.y, v1.z};
            float arr2[] = {v2.x, v2.y, v2.z};
            vector<float> pos0(arr0, arr0 + sizeof(arr0)/sizeof(arr0[0]));
            vector<float> pos1(arr1, arr1 + sizeof(arr1)/sizeof(arr1[0]));
            vector<float> pos2(arr2, arr2 + sizeof(arr2)/sizeof(arr2[0]));

            it = vertex.find(pos0);
            if (it == vertex.end()) {
                vertex[pos0] = currentIndex;
                indMap[a] = currentIndex++;
                Vec3 p0 = lower + meshLower + Vec3(v0.x, v0.y, v0.z);
                g_buffers->positions.push_back(Vec4(p0.x, p0.y, p0.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
            }
            else
            {
                indMap[a] = it->second;
            }

            it = vertex.find(pos1);
            if (it == vertex.end()) {
                vertex[pos1] = currentIndex;
                indMap[b] = currentIndex++;
                Vec3 p1 = lower + meshLower + Vec3(v1.x, v1.y, v1.z);
                g_buffers->positions.push_back(Vec4(p1.x, p1.y, p1.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
            }
            else
            {
                indMap[b] = it->second;
            }

            it = vertex.find(pos2);
            if (it == vertex.end()) {
                vertex[pos2] = currentIndex;
                indMap[c] = currentIndex++;
                Vec3 p2 = lower + meshLower + Vec3(v2.x, v2.y, v2.z);
                g_buffers->positions.push_back(Vec4(p2.x, p2.y, p2.z, invMass));
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
            }
            else
            {
                indMap[c] = it->second;
            }

            // create triangles
            g_buffers->triangles.push_back(indMap[a]);
            g_buffers->triangles.push_back(indMap[b]);
            g_buffers->triangles.push_back(indMap[c]);

            // connect springs
            
            // add spring if not duplicate
            list<uint32_t>::iterator it;
            // for a-b
            if (edgeMap.find(a) == edgeMap.end())
            {
                CreateSpring(indMap[a], indMap[b], stiffness);
                edgeMap[a].push_back(b);
            }
            else
            {

                it = find(edgeMap[a].begin(), edgeMap[a].end(), b);
                if (it == edgeMap[a].end())
                {
                    CreateSpring(indMap[a], indMap[b], stiffness);
                    edgeMap[a].push_back(b);
                }
            }

            // for a-c
            if (edgeMap.find(a) == edgeMap.end())
            {
                CreateSpring(indMap[a], indMap[c], stiffness);
                edgeMap[a].push_back(c);
            }
            else
            {

                it = find(edgeMap[a].begin(), edgeMap[a].end(), c);
                if (it == edgeMap[a].end())
                {
                    CreateSpring(indMap[a], indMap[c], stiffness);
                    edgeMap[a].push_back(c);
                }
            }

            // for b-c
            if (edgeMap.find(b) == edgeMap.end())
            {
                CreateSpring(indMap[b], indMap[c], stiffness);
                edgeMap[b].push_back(c);
            }
            else
            {

                it = find(edgeMap[b].begin(), edgeMap[b].end(), c);
                if (it == edgeMap[b].end())
                {
                    CreateSpring(indMap[b], indMap[c], stiffness);
                    edgeMap[b].push_back(c);
                }
            }
            
        }

        delete m;
    }


    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
    void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float initX = ptr[0];
        float initY = ptr[1];
        float initZ = ptr[2];
        float scale = ptr[3];
        float rot = ptr[4];
        float velX = ptr[5];
        float velY = ptr[6];
        float velZ = ptr[7];
        float stiff = ptr[8];
        float mass = ptr[9];
        float radius = ptr[10];
        cam_x = ptr[11];
        cam_y = ptr[12];
        cam_z = ptr[13];
        cam_angle_x = ptr[14];
        cam_angle_y = ptr[15];
        cam_angle_z = ptr[16];
        cam_width = int(ptr[17]);
        cam_height = int(ptr[18]);

        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        float static_friction = 0.5;
        float dynamic_friction = 1.0;

        createTshirt(make_path(tshirt_path, "/data/T-shirt_onelayer.obj"), Vec3(initX, initY, initZ), scale, rot, Vec3(velX, velY, velZ), phase, 1/mass, stiff);

        g_numSubsteps = 4;
        g_params.numIterations = 30;
        g_params.dynamicFriction = dynamic_friction;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;
        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_params.radius = radius;
        g_params.collisionDistance = 0.01f;

        g_drawPoints = true;
        g_drawMesh = false;
        g_drawCloth = true;
        g_drawSprings = false;
        g_drawDiffuse = false;

        // cout << "tris: " << g_buffers->triangles.size() << endl;
    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};