// #pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <iterator>
#include <fstream>
#include <string>


#define DIST(p, q) (sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)))
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
    char cloth_path[200];

	SoftgymTshirt(const char* name) : Scene(name) {}

    char* make_path(char* full_path, std::string path) {
        strcpy(full_path, getenv("PYFLEXROOT"));
        strcat(full_path, path.c_str());
        cout << "mesh path: " << full_path << endl;
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


    void createTshirt(const char* filename, string cloth_prefix,  Vec3 lower, float radius, float rotation, Vec3 velocity, int phase, float invMass, float stiffness, float scale)
    // Scale the cloth until the average edge length is the same as particle radius
    {

        // createTshirt(make_path(cloth_path, "/data/T-shirt_triangulated_subd.obj"), Vec3(initX, initY, initZ), 
        // scale, rot, Vec3(velX, velY, velZ), phase, 1/mass, stiff);   
        // import the mesh
        cout << "importing mesh" << endl;
        Mesh* m = ImportMesh(filename);
        if (!m) 
        {
            cout << "no mesh" << endl;
            return;
        }

        cout << "mesh faces:" << m->GetNumFaces() << endl;

        // rotate mesh
        m->Transform(RotationMatrix(rotation, Vec3(0.0f, 1.0f, 0.0f)));
        // float scale;
        float avgEdgeLen = 0.;
        //
        for (uint32_t i=0; i < m->GetNumFaces(); ++i)
        {
            // create particles
            uint32_t a = m->m_indices[i*3+0];
            uint32_t b = m->m_indices[i*3+1];
            uint32_t c = m->m_indices[i*3+2];

            Point3& v0 = m->m_positions[a];
            Point3& v1 = m->m_positions[b];
            Point3& v2 = m->m_positions[c];

            avgEdgeLen += DIST(v0, v1) + DIST(v1, v2) + DIST(v2, v0);
        }
        avgEdgeLen /= 3 * m->GetNumFaces();
        if (scale < 0){
            scale = radius / avgEdgeLen;
        }
        cout<<"Scale:"<<scale<<endl;

        Vec3 meshLower, meshUpper;
        m->GetBounds(meshLower, meshUpper);

        Matrix44 xform = ScaleMatrix(scale)*TranslationMatrix(Point3(-meshLower));
        m->Transform(xform);

        m->GetBounds(meshLower, meshUpper);

        // index of particles
        uint32_t baseIndex = uint32_t(g_buffers->positions.size());
        uint32_t currentIndex = baseIndex;

        // maps vertex by position
        // maps position to particle index
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

            //sortInd(&a, &b, &c);

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

            // TODO: normals?

            // connect springs
            
            // add spring if not duplicate
            list<uint32_t>::iterator it;
            // for a-b
            if (edgeMap.find(a) == edgeMap.end())
            {
                CreateSpring(indMap[a], indMap[b], stiffness);
//                cout<<"Spring len:"<<DIST(indMap[a], indMap[b])
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

        // save the mesh index to particle map to file
        // vertex particle
        char map_path[200];
        ofstream mapfile(make_path(map_path, "/data/" + cloth_prefix + "_map.txt"));
        map<uint32_t, uint32_t>::iterator mit;
        for (mit = indMap.begin(); mit != indMap.end(); mit++)
        {
            mapfile << mit->first << " " << mit->second << endl;
        }
        mapfile.close();

        // save the mesh index to edge map to file
        char edgemap_path[200];
        ofstream edgemapfile(make_path(edgemap_path, "/data/" + cloth_prefix +"_edgemap_id.txt"));
        map<uint32_t, list<uint32_t> >::iterator eit;
        for (eit = edgeMap.begin(); eit != edgeMap.end(); eit++)
        {
            list<uint32_t>::iterator lit;
            for (lit = eit->second.begin(); lit != eit->second.end(); lit++) 
            {
                edgemapfile << indMap[eit->first] << " " << indMap[*lit] << endl;
            }
        }
        edgemapfile.close();

        delete m;
    }


    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
    void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {
        //const char* filename = "/home/sujaybajracharya/cloth_folding/cloth_manipulation/simulation/regrasp/meshes/T-shirt_triangulated_subd.obj";
        //const char* filename = "/home/sujaybajracharya/cloth_folding/softgym/PyFlex/data/torus.obj";
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
        int render_type = int(ptr[19]);
        int cloth_type = int(ptr[20]);
        string cloth_prefix;
        if (cloth_type == 0)
            cloth_prefix = "tshirt";
        else if (cloth_type==1)
            cloth_prefix = "shorts";
        else
            cloth_prefix = "Tshirt_obj";


        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        float static_friction = 0.5;
        float dynamic_friction = 1.0;

        //CreateParticleShape(filename, Vec3(0.01, 0.15, 0.01), 0.5, 0.0f, radius, Vec3(0.0f, 0.0f, 0.0f), 0.125f, false, 1.0f, phase, true, 0.0005f, 0.0f, 0.0f, 0.0f, 0.9f);
        //createTshirtFromMesh(filename);
        cout << "creating "<<cloth_prefix << endl;
        createTshirt(make_path(cloth_path, "/data/" +  cloth_prefix + ".obj"), cloth_prefix, Vec3(initX, initY, initZ), radius, rot, Vec3(velX, velY, velZ), phase, 1/mass, stiff, scale);

        g_numSubsteps = 4;
        g_params.numIterations = 30;
        g_params.dynamicFriction = 0.75;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;
        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_params.radius = radius;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawMesh = false;
        g_drawSprings = false;
        g_drawDiffuse = false;

        // bool hasFluids = false;
        // DepthRenderProfile p = {
        //     0.f, // minRange
        //     5.f // maxRange
        // };
        // if (g_render) // ptr[19] is whether to use a depth sensor
        // {
        //     printf("adding a sensor!\n");
        //     AddSensor(cam_width, cam_height,  0,  Transform(Vec3(cam_x, cam_y, cam_z), rpy2quat(cam_angle_x, cam_angle_y, cam_angle_z)),  DegToRad(45.f), hasFluids, p);
        // }

        // DEBUG
        cout << "tris: " << g_buffers->triangles.size() << endl;
        //cout << "skinMesh: " << g_meshSkinIndices.size() << endl;
        //g_params.gravity[1] = 0.0f;
    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};