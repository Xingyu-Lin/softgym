#pragma once

#include "../core/maths.h"
#include "../external/tinyxml2/tinyxml2.h"
#include <queue>
#include <stack>

using namespace tinyxml2;
using namespace std;
#define URDF_FLATSHADE 0
#define URDF_VERBOSE 0
//#define LLL {cout<<__FILE__<<" "<<__LINE__<<endl;}
#define LLL {}
using namespace tinyxml2;

class URDFTarget 
{
public:
    URDFTarget()
    {
        jointName = "";
        compliance = 0.0f;
        angle = 0.0f;

    }
    string jointName;
    float compliance;
    float angle;
};

class URDFCollision
{
public:
    enum Type { MESH, CYLINDER, SPHERE, BOX };
    URDFCollision()
    {
        origin = Transform();
        dim = Vec3(0.0f, 0.0f, 0.0f);
        type = MESH;
        geometries.clear();
    }

    Type type;
    Transform origin;
    Vec3 dim; // for box, cylinder, sphere

    // Flex Specific
    std::vector<NvFlexRigidShape> geometries; // for mesh
};

class URDFMaterial
{
public:

    URDFMaterial() : color(0.5f)
    {
    }

    std::string name;
    Vec3 color;
};

class URDFVisual
{
public:
    enum Type { MESH, CYLINDER, SPHERE, BOX };

    URDFVisual()
    {
        origin = Transform();
        meshes.clear();
        scale = Vec3(1.0f);
    }
    Type type;
    Transform origin;
    Vec3 scale;
    Vec3 dim;
    Vec3 color;

    URDFMaterial material;

    std::vector<std::string> meshes;
};

class URDFJoint;
class URDFLink
{
public:
    URDFLink()
    {
        name = "";
        origin = Transform();
        mass = 0.0f;
        memset(inertia, 0, sizeof(float) * 9);

        inertiaValid = false;

        parentJoint = nullptr;
        childJoints.clear();
        flexBodyIndex = -1;
    }

    string name;
    Transform origin;
    float mass;
    float inertia[9];

    bool inertiaValid; // Is it real collider or visual (if visual will assume density of 1 and default moment of inertia)

    vector<URDFVisual> visuals;
    vector<URDFCollision> colliders;

    URDFJoint* parentJoint; // Parent joint if exist
    vector<URDFJoint*> childJoints; // Child joints if exists
    int flexBodyIndex; // Index in Flex's body
};

class URDFJoint
{
public:
    URDFJoint()
    {
        name = "";
        parent = NULL;
        child = NULL;
        origin = Transform();
        axis = Vec3(1.0f, 0.0f, 0.0f);
        lower = -FLT_MAX;
        upper = FLT_MAX;
        velocity = 0.0f;
        effort = 0.0f;
        damping = 0.0f;
        flexJointIndex = -1;
        mimic = "";
    }

    enum Type { FIXED, REVOLUTE, PRISMATIC, CONTINUOUS };
    Type type;
    string name;
    string mimic; // Mimic another joint?
    URDFLink* parent;
    URDFLink* child;
    Transform origin;
    Vec3 axis;

    float lower, upper;
    float velocity; // Velocity limit
    float damping;
    float effort; // Force limit

    int flexJointIndex; // Index in Flex's joint
};

class URDFCableLink
{
public:
    URDFCableLink()
    {
        type = FIXED;
        body = 0;
        in = out = Vec3(0.0f, 0.0f, 0.0f);
        extraLength = 0.0f;
    }

    enum Type { FIXED, PINHOLE, ROLLING };
    Type type;
    URDFLink* body;

    union
    {
        // For fixed
        Vec3 pos;

        // For rolling
        Vec3 normal;

        // For pinhole
        Vec3 in;
    };

    union
    {
        // For rolling
        float d;

        // For pinhole
        Vec3 out;
    };

    // For rolling
    float extraLength;

};

class URDFCable
{
public:
    URDFCable()
    {
        cyclic = false;
        name = "";
        stretchingCompliance = 0.0f;
        compressionCompliance = 0.0f;
    }
    bool cyclic;
    string name;
    vector<URDFCableLink> links;
    float stretchingCompliance;
    float compressionCompliance;
};

class URDFImporter
{
public:
    vector<URDFTarget> targets;
    vector<URDFJoint*> joints;
    vector<URDFLink*> links;
    vector<URDFCable*> cables;

    map<URDFLink*, int> rigidMap;

    map<string, int> urdfJointNameMap;		// Map from name to the index of joint in URDF file
    map<string, int> rigidNameMap;          // Map from name to the index of rigid body
    map<string, pair<int, int> > rigidShapeNameMap;     // Map from name to the index of rigid body' shape
    map<string, NvFlexCurveId> rigidCurveMap;			// Map from name to curve ID
    map<string, int> jointNameMap;          // Map from name to the index of limiting joint
    map<string, int> activeJointNameMap;    // Map from name to the index of position target joint
    map<string, int> jointDOFNameMap;       // Map from name to the DOF of joint

    map<string, URDFMaterial> materialsNameMap;	// Map from name to link material

    bool replaceCylinderWithCapsule;
    int slicesPerCylinder;
    bool useSphereIfNoCollision; //If there is no collision section in URDF use a small sphere
    map<string, std::pair<Mesh*, RenderMesh*> > meshCache;	
    map<pair<float, float>, NvFlexTriangleMeshId> cylinderCache;


    tinyxml2::XMLDocument doc;


    class FloatPairComp
    {
    public:
        bool operator () (const pair<float, float>& a, const pair<float, float>& b) const
        {
            if (a.first < b.first) return true;
            if (a.first > b.first) return false;
            return a.second < b.second;
        }
    };

    void loadConvexesFromObj(const char* fname, const Vec3& scale, std::vector<NvFlexRigidShape>& geometries, float dilation, float thickness)
    {
        Mesh* mesh = ImportMeshFromObj(fname);
        mesh->Transform(ScaleMatrix(scale));

        // use flat normals on collision shapes
        mesh->CalculateFaceNormals();

        NvFlexRigidShape shape;
        NvFlexTriangleMeshId trimesh = CreateTriangleMesh(mesh, dilation);
        NvFlexMakeRigidTriangleMeshShape(&shape, 0, trimesh, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()), 1.0f, 1.0f, 1.0f);
        shape.thickness = thickness;

        geometries.push_back(shape);

    }

    void loadInlineMesh(XMLElement* mesh, vector<Vec3>& verts, vector<int>& indices)
    {
        XMLElement* vs = mesh->FirstChildElement("vertices");
        int numV = atoi(vs->Attribute("count"));
        verts.resize(numV);
        XMLElement* v = vs->FirstChildElement("vertex");
        for (int i = 0; i < numV; i++)
        {
            int id = atoi(v->Attribute("id"));
            sscanf(v->Attribute("xyz"), "%f %f %f", &verts[id].x, &verts[id].y, &verts[id].z);
            v = v->NextSiblingElement("vertex");
        }
        XMLElement* fs = mesh->FirstChildElement("faces");
        XMLElement* f = fs->FirstChildElement("face");
        vector<int> inds;
        while (f)
        {
            istringstream ins(f->Attribute("indices"));
            inds.clear();
            int id;
            while (ins >> id)
            {
                inds.push_back(id);
            }
            for (int i = 0; i < inds.size() - 2; i++)
            {
                indices.push_back(inds[0]);
                indices.push_back(inds[i + 1]);
                indices.push_back(inds[i + 2]);
            }
            f = f->NextSiblingElement("face");
        }
    }

    Mesh* getMeshFromInlineMesh(XMLElement* m)
    {
        Mesh* mesh = new Mesh();
        vector<Vec3> verts;
        vector<int> indices;
        loadInlineMesh(m, verts, indices);
        for (int i = 0; i < (int)verts.size(); i++)
        {
            mesh->m_positions.push_back(Point3(verts[i].x, verts[i].y, verts[i].z));
        }
        for (int i = 0; i < (int)indices.size(); i++)
        {
            mesh->m_indices.push_back(indices[i]);
        }
        mesh->CalculateNormals();
        return mesh;
    }

    void loadTriMeshesFromInlineMesh(XMLElement* m, std::vector<NvFlexRigidShape>& geometries, float dilation, float thickness)
    {
        Mesh* mesh = getMeshFromInlineMesh(m);

        NvFlexRigidShape shape;
        NvFlexTriangleMeshId trimesh = CreateTriangleMesh(mesh, dilation);
        NvFlexMakeRigidTriangleMeshShape(&shape, 0, trimesh, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()), 1.0f, 1.0f, 1.0f);
        shape.thickness = thickness;

        geometries.push_back(shape);

    }

    void loadConvexesFromWrl(const char* fname, float scale, std::vector<NvFlexRigidShape>& geometries, float dilation, float thickness)
    {
        ifstream inf(fname);
        if (!inf)
        {
            printf("File %s not found!\n", fname);
        }

        string str;
        while (inf >> str)
        {
            if (str == "point")
            {
                vector<Vec3> points;
                //printf("Found convex\n");
                string tmp;
                inf >> tmp;
                while (tmp != "]")
                {
                    float x, y, z;
                    string ss;
                    inf >> ss;
                    if (ss == "]")
                    {
                        break;
                    }
                    x = (float)atof(ss.c_str());
                    inf >> y >> z;
                    //printf("      %f %f %f\n", x, y, z);
                    points.push_back(Vec3(x, y, z) * scale);
                    inf >> tmp;
                }

                //printf("convex with %d points\n", (int)points.size());
                while (inf >> str)
                {
                    if (str == "coordIndex")
                    {
                        //printf("Found coordIndex\n");
                        vector<int> indices;
                        inf >> tmp;
                        //cout << tmp << endl;
                        inf >> tmp;
                        while (tmp != "]")
                        {
                            int i0, i1, i2;

                            if (tmp == "]")
                            {
                                break;
                            }
                            sscanf(tmp.c_str(), "%d", &i0);
                            string s1, s2, s3;
                            inf >> s1 >> s2 >> s3;
                            sscanf(s1.c_str(), "%d", &i1);
                            sscanf(s2.c_str(), "%d", &i2);
                            indices.push_back(i0);
                            indices.push_back(i1);
                            indices.push_back(i2);
                            inf >> tmp;
                        }

                        //printf("convex with %d indices\n", (int)indices.size());
                        // Now found triangles too, create Flex convex
                        Mesh mesh;
                        mesh.m_positions.resize(points.size());
                        mesh.m_indices.resize(indices.size());
                        for (size_t i = 0; i < points.size(); i++)
                        {
                            mesh.m_positions[i].x = points[i].x;
                            mesh.m_positions[i].y = points[i].y;
                            mesh.m_positions[i].z = points[i].z;
                        }
                        memcpy(&mesh.m_indices[0], &indices[0], sizeof(int)*indices.size());
                        mesh.CalculateNormals();
                        NvFlexRigidShape shape;
                        /*
                           NvFlexConvexMeshId convex = CreateConvexMesh(&mesh);
                           NvFlexMakeRigidConvexMeshShape(&shape, 0, convex, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()), 1.0f, 1.0f, 1.0f);
                           */

                        NvFlexTriangleMeshId trimesh = CreateTriangleMesh(&mesh, dilation);
                        NvFlexMakeRigidTriangleMeshShape(&shape, 0, trimesh, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()), 1.0f, 1.0f, 1.0f);
                        shape.thickness = thickness;

                        geometries.push_back(shape);

                        break;
                    }
                }
            }
        }
        inf.close();
    }

    Vec3 getVec3(const char* st)
    {
        if (!st)
        {
            return Vec3(0.0f, 0.0f, 0.0f);
        }
        Vec3 p;
        sscanf(st, "%f %f %f", &p.x, &p.y, &p.z);
        return p;
    }


    Quat getQuat(const char* st)
    {
        if (!st)
        {
            return Quat(0.0f, 0.0f, 0.0f, 1.0f);
        }
        Quat q;
        sscanf(st, "%f %f %f %f", &q.x, &q.y, &q.z, &q.w);
        return q;
    }

    Transform getTransform(XMLElement* e, const char* name)
    {
        if (!e)
        {
            return Transform();
        }

        XMLElement* c = e->FirstChildElement(name);
        if (!c)
        {
            return Transform();
        }

        Vec3 p = getVec3(c->Attribute("xyz"));
        //Vec3 rpy = getVec3(c->Attribute("rpy"));
        Quat q;
        if (c->Attribute("quat"))
        {
            q = getQuat(c->Attribute("quat"));
        }
        else
        {
            Vec3 rpy = getVec3(c->Attribute("rpy"));
            q = rpy2quat(rpy.x, rpy.y, rpy.z);
        }
        return Transform(p, q);
    }

    float getFloat(XMLElement* e, const char* name, const char* aname)
    {
        XMLElement* c = e->FirstChildElement(name);
        string st = c->Attribute(aname);
        float t;
        sscanf(st.c_str(), "%f", &t);

        return t;
    }

    float getFloat(XMLElement* e, const char* name)
    {
        return getFloat(e, name, "value");
    }

    void getFloatIfExist(XMLElement* e, const char* aname, float& v)
    {
        if (e->Attribute(aname))
        {
            v = (float)atof(e->Attribute(aname));
        }
    }

    Vec3 getColor(XMLElement* e, const char* name)
    {
        XMLElement* c = e->FirstChildElement(name);
        if (!c)
        {
            return Vec3(0.0f, 0.0f, 0.0f);
        }
        string st = c->Attribute("rgba");
        return getVec3(c->Attribute("rgba"));
    }


    Vec3 getVec3(XMLElement* e, const char* name)
    {
        XMLElement* c = e->FirstChildElement(name);
        if (!c)
        {
            return Vec3(0.0f, 0.0f, 0.0f);
        }
        string st = c->Attribute("xyz");
        return getVec3(c->Attribute("xyz"));
    }

    void getInertia(XMLElement* e, const char* name, float* in)
    {
        XMLElement* c = e->FirstChildElement(name);
        float ixx = 0.0f, ixy = 0.0f, ixz = 0.0f, iyy = 0.0f, iyz = 0.0f, izz = 0.0f;
        ixx = (float)atof(c->Attribute("ixx"));
        ixy = (float)atof(c->Attribute("ixy"));
        ixz = (float)atof(c->Attribute("ixz"));
        iyy = (float)atof(c->Attribute("iyy"));
        iyz = (float)atof(c->Attribute("iyz"));
        izz = (float)atof(c->Attribute("izz"));

        in[0] = ixx;
        in[1] = ixy;
        in[2] = ixz;
        in[3] = ixy;
        in[4] = iyy;
        in[5] = iyz;
        in[6] = ixz;
        in[7] = iyz;
        in[8] = izz;
    }

    void getLimit(URDFJoint* j, XMLElement* joint)
    {
        XMLElement* lim = joint->FirstChildElement("limit");
        if (!lim)
        {
            return;
        }
        getFloatIfExist(lim, "effort", j->effort);
        getFloatIfExist(lim, "lower", j->lower);
        getFloatIfExist(lim, "upper", j->upper);
        getFloatIfExist(lim, "velocity", j->velocity);
    }

    void traverseAndTransform(URDFJoint* joint, Transform tran)
    {
        joint->origin = tran * joint->origin;
        if (joint->child)
        {
            traverseAndTransform(joint->child, joint->origin);
        }

    }

    void traverseAndTransform(URDFLink* link, Transform tran)
    {
        link->origin = tran * link->origin;

        // Not sure!
        for (int i = 0; i < (int)link->colliders.size(); i++)
        {
            link->colliders[i].origin = tran * link->colliders[i].origin;
        }

        for (int i = 0; i < (int)link->visuals.size(); i++)
        {
            link->visuals[i].origin = tran * link->visuals[i].origin;
        }

        for (int i = 0; i < (int)link->childJoints.size(); i++)
        {
            traverseAndTransform(link->childJoints[i], tran);
        }
    }
    void setTransform(XMLElement* e, const char* name, const Transform& t)
    {
        if (!e)
        {
            return;
        }

        XMLElement* c = e->FirstChildElement(name);
        if (!c)
        {
            //return;
            XMLElement* t = doc.NewElement(name);
            e->InsertFirstChild(t);
            c = e->FirstChildElement(name);
        }

        char tmp[500];
        sprintf(tmp, "%f %f %f", t.p.x, t.p.y, t.p.z);
        c->SetAttribute("xyz", tmp);

        sprintf(tmp, "%f %f %f %f", t.q.x, t.q.y, t.q.z, t.q.w);
        c->SetAttribute("quat", tmp);
        c->DeleteAttribute("rpy"); // Remove roll pitch yaw
        /*
           double roll, pitch, yaw;
           quat2rpy(t.q, roll, pitch, yaw);
           sprintf(tmp, "%lf %lf %lf", roll, pitch, yaw);
           c->SetAttribute("rpy", tmp);
           */
    }

    void setFloat(XMLElement* e, const char* name, const char* aname, float val)
    {
        XMLElement* c = e->FirstChildElement(name);
		if (!c) 
		{
			XMLElement* t = doc.NewElement(name);
			e->InsertFirstChild(t);
			c = e->FirstChildElement(name);
		}

        c->SetAttribute(aname, val);
    }

    void setFloat(XMLElement* e, const char* name, float val)
    {

        return setFloat(e, name, "value", val);
    }

    void getInertial(XMLElement* inertial, Transform& tran, float& mass, Matrix33& inertia)
    {
        tran = getTransform(inertial, "origin");
        mass = getFloat(inertial, "mass");
        getInertia(inertial, "inertia", (float*)&inertia);
    }

    void setInertia(XMLElement* e, const char* name, float* in)
    {
        XMLElement* c = e->FirstChildElement(name);
		if (!c) 
		{
			XMLElement* t = doc.NewElement(name);
			e->InsertFirstChild(t);
			c = e->FirstChildElement(name);
		}
        c->SetAttribute("ixx", in[0]);
        c->SetAttribute("ixy", in[1]);
        c->SetAttribute("ixz", in[2]);
        c->SetAttribute("iyy", in[4]);
        c->SetAttribute("iyz", in[7]);
        c->SetAttribute("izz", in[8]);
    }

    void setInertial(XMLElement* inertial, const Transform& tran, const float mass, const Matrix33& inertia)
    {
        setTransform(inertial, "origin", tran);
        setFloat(inertial, "mass", mass);
        setInertia(inertial, "inertia", (float*)&inertia);
    }

    void LumpFixedJointsAndSaveURDF(const char* oname)
    {
        // First, build hierarchy of bodies linked by fixed joint
        map<string, pair<int, XMLElement*> > linkMap;
        map<string, pair<int, XMLElement*> > jointMap;

        // --- Build map of link's name -> pair(link's index, XMLElement*)
        for (int i = 0; i < (int)links.size(); i++)
        {
            linkMap[links[i]->name].first = i;
        }
        XMLElement* root = doc.RootElement();
        XMLElement* link = root->FirstChildElement("link");
        while (link)
        {
            linkMap[link->Attribute("name")].second = link;
            link = link->NextSiblingElement("link");
        }
        // --- Build map of joint's name -> pair(joint's index, XMLElement*)
        for (int i = 0; i < (int)joints.size(); i++)
        {
            jointMap[joints[i]->name].first = i;
        }
        XMLElement* joint = root->FirstChildElement("joint");
        while (joint)
        {
            jointMap[joint->Attribute("name")].second = joint;
            joint = joint->NextSiblingElement("joint");
        }

        vector<vector<int> > fixedChildren(links.size());
        vector<vector<int> > fixedChildrenJoint(links.size());
        vector<vector<int> > allChildrenJoint(links.size());
        vector<bool> involve(links.size(), false);
        vector<bool> isRoot(links.size(), true);
        // Do DFS to identify depth from non-fixed body
        for (int i = 0; i < (int)joints.size(); i++)
        {
            if (joints[i]->type == URDFJoint::FIXED)
            {
                fixedChildren[linkMap[joints[i]->parent->name].first].push_back(linkMap[joints[i]->child->name].first);
                fixedChildrenJoint[linkMap[joints[i]->parent->name].first].push_back(i);
                involve[linkMap[joints[i]->parent->name].first] = involve[linkMap[joints[i]->child->name].first] = true;
                isRoot[linkMap[joints[i]->child->name].first] = false;
            }
            allChildrenJoint[linkMap[joints[i]->parent->name].first].push_back(i);
        }

        stack<int> mergeOrder;
        vector<int> depth(links.size(), (int)links.size() + 100);
        // Root is what's involve and still have root = true
        for (int i = 0; i < (int)links.size(); i++)
        {
            if (involve[i] && isRoot[i])
            {
                // Do BFS
                queue<int> q;
                q.push(i);
                depth[i] = 0;
                //mergeOrder.push(i); // This is root, no need to merge
                while (!q.empty())
                {
                    int l = q.front();
                    q.pop();
                    for (int j = 0; j < (int)fixedChildren[l].size(); j++)
                    {
                        depth[fixedChildren[l][j]] = depth[l] + 1;
                        q.push(fixedChildren[l][j]);
                        mergeOrder.push(fixedChildrenJoint[l][j]);
                    }
                }
            }
        }

        // Popping mergeOrder from stack will give the correct ordering for merging bodies
		while (!mergeOrder.empty())
		{
			int j = mergeOrder.top();
			mergeOrder.pop();

			// Merge fixed body to its parent, lump moment of inertia to parent, move visual and collision to parent with modified origin, delete fixed joint
			URDFJoint* joint = joints[j];

			Transform jorig = getTransform(jointMap[joint->name].second, "origin");

			// Lump inertia to parent
			Transform ptrans;
			float pmass;
			Matrix33 pinertia;
			if (linkMap[joint->parent->name].second->FirstChildElement("inertial")) 
			{
				getInertial(linkMap[joint->parent->name].second->FirstChildElement("inertial"), ptrans, pmass, pinertia);
			}
			else {
				// Obtain from body
				NvFlexRigidBody& rb = g_buffers->rigidBodies[rigidNameMap[joint->parent->name]];
				pmass = rb.mass;
				pinertia = Matrix33(rb.inertia);
			}

			Transform ctrans;
			float cmass;
			Matrix33 cinertia;
			if (!linkMap[joint->child->name].second) 
			{
				cout << "Child link doesn't exist" << endl;
			}
			if (linkMap[joint->child->name].second->FirstChildElement("inertial")) 
			{
				getInertial(linkMap[joint->child->name].second->FirstChildElement("inertial"), ctrans, cmass, cinertia);
			}
			else {
				// Obtain from body
				NvFlexRigidBody& rb = g_buffers->rigidBodies[rigidNameMap[joint->child->name]];
				cmass = rb.mass;
				cinertia = Matrix33(rb.inertia);
			}

			// Express inertia of children in parent coordinate
			ctrans = jorig*ctrans;

			float newMass = pmass + cmass;
			Vec3 newCenter = (ptrans.p*pmass + ctrans.p*cmass) / newMass;
			Matrix33 ctransM(ctrans.q);
			Matrix33 ptransM(ptrans.q);
			Vec3 coffset = ctrans.p - newCenter;
			Vec3 poffset = ptrans.p - newCenter;
			Matrix33 newInertia =
				Transpose(ptransM)*(
					ptransM*pinertia*Transpose(ptransM) + pmass*(LengthSq(poffset)*Matrix33::Identity() - Outer(poffset, poffset)) +
					ctransM*cinertia*Transpose(ctransM) + cmass*(LengthSq(coffset)*Matrix33::Identity() - Outer(coffset, coffset))
					)*ptransM; // Express in the p's reference frame
			Quat newQuat = ptrans.q;
			if (!linkMap[joint->parent->name].second->FirstChildElement("inertial")) 
			{
				XMLElement* t = doc.NewElement("inertial");
				linkMap[joint->parent->name].second->InsertFirstChild(t);
			}
            setInertial(linkMap[joint->parent->name].second->FirstChildElement("inertial"), Transform(newCenter, newQuat), newMass, newInertia);

            // Move colliders of children to parent
            XMLElement* col = linkMap[joint->child->name].second->FirstChildElement("collision");
            while (col)
            {
                XMLElement* ocol = col;
                Transform trans = getTransform(col, "origin");
                trans = jorig*trans;
                setTransform(col, "origin", trans);
                col = col->NextSiblingElement("collision");
                linkMap[joint->parent->name].second->InsertEndChild(ocol); // Move to parent
            }

            // Move visuals of children to parent
            XMLElement* vis = linkMap[joint->child->name].second->FirstChildElement("visual");
            while (vis)
            {
                XMLElement* ovis = vis;
                Transform trans = getTransform(vis, "origin");
                trans = jorig*trans;
                setTransform(vis, "origin", trans);
                vis = vis->NextSiblingElement("visual");
                linkMap[joint->parent->name].second->InsertEndChild(ovis); // Move to parent
            }

            // Change all joints connected to children to instead connect to parent
            int childLinkIndex = linkMap[joint->child->name].first;
            for (int i = 0; i < (int)allChildrenJoint[childLinkIndex].size(); i++)
            {
                URDFJoint* cj = joints[allChildrenJoint[childLinkIndex][i]];
                XMLElement* jo = jointMap[cj->name].second;
                Transform trans = getTransform(jo, "origin");

                trans = jorig*trans;
                setTransform(jo, "origin", trans);

                Transform ntrans;
                ntrans = getTransform(jo, "origin");
                //printf("Set parent of joint %s to %s\n", cj->name.c_str(), joint->parent->name.c_str());
                if ((Length(ntrans.p - trans.p) > 1e-5f) || (Length(ntrans.q - trans.q) > 1e-5f)) 
				{
                    cout << "Convert quat -> rpy -> quat, does not exactly the same result!" << endl;
                }
                jo->FirstChildElement("parent")->SetAttribute("link", joint->parent->name.c_str());
            }

            // Remove this joint
            doc.DeleteNode(jointMap[joint->name].second);

            // Remove this body
            doc.DeleteNode(linkMap[joint->child->name].second);
        }
		
		// Save URDF
        doc.SaveFile(oname);
    }

    URDFImporter(const string rootPath /*replace package:// */,
                 const string urdfFileRelativeToRoot, bool useObj=false, float dilation=0.005f, float thickness=0.005f, bool replaceCylinderWithCapsule=true, int slicesPerCylinder = 20, bool useSphereIfNoCollision = true)
    {
        this->replaceCylinderWithCapsule = replaceCylinderWithCapsule;
        this->slicesPerCylinder = slicesPerCylinder;
        this->useSphereIfNoCollision = useSphereIfNoCollision;
        LLL;
        LLL;

        std::string path;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
        path = GetFilePathByPlatform((rootPath + "\\" + urdfFileRelativeToRoot).c_str());
#else
        path = GetFilePathByPlatform((rootPath + "/" + urdfFileRelativeToRoot).c_str());
#endif
#if URDF_VERBOSE
        cout << "Load file: " << path << endl;
#endif
        doc.LoadFile(path.c_str());

        LLL;
        XMLElement* root = doc.RootElement();

        if (!root)
        {
            cout << "Could not load URDF " << path << endl;
            return;
        }

        // Load all materials
        XMLElement* material = root->FirstChildElement("material");

        if (!material)
        {
            // do nothing
        }
        else
        {
            do
            {
                string name = material->Attribute("name");

                // if there is already a material with this name then read it
                URDFMaterial& mat = materialsNameMap[name];

                // update it
                mat.name = name;

                if (material->FirstChildElement("color"))
                    mat.color = getColor(material, "color");
                      
            } while ((material = material->NextSiblingElement("material")));
        }

        LLL;
        map<string, URDFLink*> linkH;
        LLL;
        // Load all links
        XMLElement* link = root->FirstChildElement("link");
        if (!link)
        {
#if URDF_VERBOSE
            cout << "Link not found!" << endl;
#endif
        }
        do
        {
#if URDF_VERBOSE
            cout << "Link " << link->Attribute("name") << endl;
#endif
            XMLElement* inertial = link->FirstChildElement("inertial");
            URDFLink* ul = new URDFLink();
            ul->name = link->Attribute("name");
            if (inertial)
            {
                ul->origin = getTransform(inertial, "origin");
                //printf("origin pos = %f %f %f, quat = %f %f %f %f\n", ul->origin.p.x, ul->origin.p.y, ul->origin.p.z, ul->origin.q.x, ul->origin.q.y, ul->origin.q.z, ul->origin.q.w);
                ul->mass = getFloat(inertial, "mass");
                getInertia(inertial, "inertia", ul->inertia);

                ul->inertiaValid = true;
            }

            XMLElement* collision = link->FirstChildElement("collision");
            if (!collision)
            {
                if (useSphereIfNoCollision)
                {
#if URDF_VERBOSE
                    printf("Link with inertia but without collision!, assume to be a small sphere\n");
#endif
                    ul->colliders.resize(1);
                    ul->colliders.back().origin = ul->origin;
                    ul->colliders.back().dim.x = 0.01f;
                    ul->colliders.back().type = URDFCollision::SPHERE;
                }
                else
                {
#if URDF_VERBOSE
                    printf("Link with inertia but without collision!\n");
#endif
                }
            }
            else
            {
                // There is a collision
                while (collision)
                {
                    ul->colliders.push_back(URDFCollision());
                    ul->colliders.back().origin = getTransform(collision, "origin");
                    XMLElement* geometry = collision->FirstChildElement("geometry");
                    XMLElement* e;
                    if ((e = geometry->FirstChildElement("mesh")))
                    {
                        // Mesh
                        if (e->Attribute("filename"))
                        {
                            string fname = e->Attribute("filename");
                            string pack = "package://";
                            size_t p = fname.find(pack);
                            if (p != fname.npos)
                            {
                                fname.replace(p, pack.size(), "");
                            }

                            if (!useObj)
                            {
                                string stl = ".stl";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".wrl");
                                }
                                stl = ".STL";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".wrl");
                                }

                                // replace objs with wrls
                                stl = ".obj";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".wrl");
                                }
                                stl = ".OBJ";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".wrl");
                                }

                                loadConvexesFromWrl((rootPath + "/" + fname).c_str(), 1.0f, ul->colliders.back().geometries, dilation, thickness);

                            }
                            else
                            {
                                string stl = ".stl";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".obj");
                                }
                                stl = ".STL";
                                p = fname.find(stl);
                                if (p != fname.npos)
                                {
                                    fname.replace(p, stl.size(), ".obj");
                                }

                                Vec3 scale(1.0f);
                                if (e->Attribute("scale"))
                                {
                                    scale = getVec3(e->Attribute("scale"));
                                }

                                loadConvexesFromObj((rootPath + "/" + fname).c_str(), scale, ul->colliders.back().geometries, dilation, thickness);
                            }
#if URDF_VERBOSE
                            cout << "Load convexes from " << fname << " has " << ul->colliders.back().geometries.size() << " geometries" << endl;
#endif
                        }
                        else
                        {
                            // Extension with built in vertices
                            loadTriMeshesFromInlineMesh(e, ul->colliders.back().geometries, dilation, thickness);
                        }
                        ul->colliders.back().type = URDFCollision::MESH;
                    }
                    else if ((e = geometry->FirstChildElement("box")))
                    {
                        // box with size
                        ul->colliders.back().dim = getVec3(e->Attribute("size"));
                        //printf("Box of dim %f %f %f\n", ul->collider.dim.x, ul->collider.dim.y, ul->collider.dim.z);
                        ul->colliders.back().type = URDFCollision::BOX;
                    }
                    else if ((e = geometry->FirstChildElement("sphere")))
                    {
                        // sphere with radius
                        ul->colliders.back().dim.x = (float)atof(e->Attribute("radius"));
                        ul->colliders.back().type = URDFCollision::SPHERE;
                    }
                    else if ((e = geometry->FirstChildElement("cylinder")))
                    {
                        // cylinder with length and radius
                        ul->colliders.back().dim.x = (float)atof(e->Attribute("radius"));
                        ul->colliders.back().dim.y = (float)atof(e->Attribute("length"));
                        ul->colliders.back().type = URDFCollision::CYLINDER;
                        if (replaceCylinderWithCapsule)
                        {
                            ul->colliders.back().origin.q = ul->colliders.back().origin.q*QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f);								
                        }
                        else
                        {
                            ul->colliders.back().origin.q = ul->colliders.back().origin.q*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi*0.5f);
                        }

                        //ul->collider.origin.q = Quat(kPi*0.5f, Vec3(1.0f, 0.0f, 0.0f));
                    }
                    collision = collision->NextSiblingElement("collision");
                }
            }


            if (ul->colliders.size() == 0)
            {
                // If no collider but there's visual, use visual as collision
                XMLElement* visual = link->FirstChildElement("visual");
                LLL;

                while (visual)
                {
                    ul->colliders.push_back(URDFCollision());
                    ul->colliders.back().type = URDFCollision::MESH;
                    ul->mass = FLT_MAX;
                    ul->origin = ul->colliders.back().origin = getTransform(visual, "origin");
                    XMLElement* geometry = visual->FirstChildElement("geometry");
                    XMLElement* e;

                    if ((e = geometry->FirstChildElement("mesh")))
                    {
                        // Mesh
                        if (e->Attribute("filename"))
                        {
                            string fname = e->Attribute("filename");
                            string pack = "package://";
                            size_t p = fname.find(pack);
                            if (p != fname.npos)
                            {
                                fname.replace(p, pack.size(), "");
                            }

                            string stl = ".stl";
                            p = fname.find(stl);
                            if (p != fname.npos)
                            {
                                fname.replace(p, stl.size(), ".wrl");
                            }

                            stl = ".STL";
                            p = fname.find(stl);
                            if (p != fname.npos)
                            {
                                fname.replace(p, stl.size(), ".wrl");
                            }

                            loadConvexesFromWrl((rootPath + "/" + fname).c_str(), 1.0f, ul->colliders.back().geometries, dilation, thickness);
                        }
                        else
                        {
                            // Extension with built in vertices
                            loadTriMeshesFromInlineMesh(e, ul->colliders.back().geometries, dilation, thickness);
                        }
                        ul->colliders.back().type = URDFCollision::MESH;
                    }
                    else if ((e = geometry->FirstChildElement("box")))
                    {
                        // box with size
                        ul->colliders.back().dim = getVec3(e->Attribute("size"));
                        ul->colliders.back().type = URDFCollision::BOX;
                    }
                    else if ((e = geometry->FirstChildElement("sphere")))
                    {
                        // sphere with radius
                        ul->colliders.back().dim.x = (float)atof(e->Attribute("radius"));
                        ul->colliders.back().type = URDFCollision::SPHERE;
                    }
                    else if ((e = geometry->FirstChildElement("cylinder")))
                    {
                        // cylinder with length and radius
                        ul->colliders.back().dim.x = (float)atof(e->Attribute("radius"));
                        ul->colliders.back().dim.y = (float)atof(e->Attribute("length"));
                        ul->colliders.back().type = URDFCollision::CYLINDER;

                        ul->colliders.back().origin.q = ul->colliders.back().origin.q * QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f);
                        //ul->collider.origin.q = Quat(kPi*0.5f, Vec3(1.0f, 0.0f, 0.0f));
                    }
                    visual = visual->NextSiblingElement("visual");
                }
            }

            XMLElement* visual = link->FirstChildElement("visual");
            while (visual)
            {
                ul->visuals.push_back(URDFVisual());
                ul->visuals.back().origin = getTransform(visual, "origin");

                XMLElement* m;
                if ((m = visual->FirstChildElement("material")))
                {
                    string name = m->Attribute("name");

                    // if there is already a material with this name then read it
                    URDFMaterial& mat = materialsNameMap[name];

                    // update it
                    mat.name = name;

                    if (m->FirstChildElement("color"))
                        mat.color = getColor(m, "color");

                    ul->visuals.back().material = mat;
                }


                XMLElement* geometry = visual->FirstChildElement("geometry");
                XMLElement* e;
                if ((e = geometry->FirstChildElement("mesh")))
                {
                    if (e->Attribute("scale"))
                    {
                        ul->visuals.back().scale = getVec3(e->Attribute("scale"));
                    }

                    // Mesh
                    if (e->Attribute("filename"))
                    {
                        // Mesh
                        string fname = e->Attribute("filename");
                        string pack = "package://";
                        size_t p = fname.find(pack);
                        if (p != fname.npos)
                        {
                            fname.replace(p, pack.size(), "");
                        }

                        string stl = ".stl";
                        p = fname.find(stl);
                        if (p != fname.npos)
                        {
                            fname.replace(p, stl.size(), ".wrl");
                        }

                        stl = ".STL";
                        p = fname.find(stl);
                        if (p != fname.npos)
                        {
                            fname.replace(p, stl.size(), ".wrl");
                        }

                        string dae = ".dae";
                        p = fname.find(dae);
                        if (p != fname.npos)
                        {
                            fname.replace(p, dae.size(), ".obj");
                        }

                        dae = ".DAE";
                        p = fname.find(dae);
                        if (p != fname.npos)
                        {
                            fname.replace(p, dae.size(), ".obj");
                        }


                        ul->visuals.back().meshes.push_back((rootPath + "/" + fname));
                    }
                    else
                    {
                        // Put in mesh cache directly
                        Mesh* mesh = getMeshFromInlineMesh(e);

                        mesh->Transform(ScaleMatrix(ul->visuals.back().scale));

                        string meshName = "###" + to_string(meshCache.size());

                        meshCache[meshName] = make_pair(mesh, CreateRenderMesh(mesh));
                        ul->visuals.back().meshes.push_back(meshName);
                    }
                    if (e->Attribute("scale"))
                    {
                        ul->visuals.back().scale = getVec3(e->Attribute("scale"));
                    }
                    // load mesh
                    ul->visuals.back().type = URDFVisual::MESH;                    
                }
                else if ((e = geometry->FirstChildElement("box")))
                {
                    // box with size
                    ul->visuals.back().dim = getVec3(e->Attribute("size"));
                    ul->visuals.back().type = URDFVisual::BOX;
                }
                else if ((e = geometry->FirstChildElement("sphere")))
                {
                    // sphere with radius
                    ul->visuals.back().dim.x = (float)atof(e->Attribute("radius"));
                    ul->visuals.back().type = URDFVisual::SPHERE;
                }
                else if ((e = geometry->FirstChildElement("cylinder")))
                {
                    // cylinder with length and radius
                    ul->visuals.back().dim.x = (float)atof(e->Attribute("radius"));
                    ul->visuals.back().dim.y = (float)atof(e->Attribute("length"));
                    ul->visuals.back().type = URDFVisual::CYLINDER;

                    ul->visuals.back().origin.q = ul->visuals.back().origin.q * QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f);
                    //ul->collider.origin.q = Quat(kPi*0.5f, Vec3(1.0f, 0.0f, 0.0f));
                }
                visual = visual->NextSiblingElement("visual");
            }

            linkH[link->Attribute("name")] = ul;
            links.push_back(ul);
        }
        while ((link = link->NextSiblingElement("link")));
        LLL;

        // Load all joint
        XMLElement* joint = root->FirstChildElement("joint");
        if (joint)
        {
            do
            {
                URDFJoint* uj = new URDFJoint();
                string type = joint->Attribute("type");
                uj->name = joint->Attribute("name");
                uj->origin = getTransform(joint, "origin");
                printf("Joint %s\n", uj->name.c_str());
                printf("origin pos = %f %f %f, quat = %f %f %f %f\n", uj->origin.p.x, uj->origin.p.y, uj->origin.p.z, uj->origin.q.x, uj->origin.q.y, uj->origin.q.z, uj->origin.q.w);

                uj->parent = linkH[joint->FirstChildElement("parent")->Attribute("link")];
                uj->child = linkH[joint->FirstChildElement("child")->Attribute("link")];

                if (joint->FirstChildElement("mimic"))
                {
                    uj->mimic = joint->FirstChildElement("mimic")->Attribute("joint");
                }
                if (uj->child)
                {
                    uj->child->parentJoint = uj;
                }

                if (!uj->parent || !uj->child)
                {
                    printf("Parent or child not found!!!!!\n");
                    //exit(0);
                }

				uj->parent->childJoints.push_back(uj);
                uj->origin = getTransform(joint, "origin");

                if (type == "fixed")
                {
                    // Fixed joint
                    uj->type = URDFJoint::FIXED;
                }
                else if ((type == "prismatic") || (type == "revolute") || (type == "continuous"))
                {
                    // Prismatic
                    uj->axis = getVec3(joint, "axis");
                    if (joint->FirstChildElement("dynamics"))
                    {
                        uj->damping = (float)atof(joint->FirstChildElement("dynamics")->Attribute("damping"));
                    }
                    getLimit(uj, joint);
                    if (type == "prismatic")
                    {
                        uj->type = URDFJoint::PRISMATIC;
                    }
                    else if (type == "revolute")
                    {
                        uj->type = URDFJoint::REVOLUTE;
                    }
                    else
                    {
                        uj->type = URDFJoint::CONTINUOUS;
                    }
                }
                else
                {
                    printf("Unsupported joint type %s\n", type.c_str());
                    //exit(0);
                }

                if (joint->FirstChildElement("target"))
                {
                    // Read target, if there's one
                    XMLElement* tar = joint->FirstChildElement("target");
                    URDFTarget target;
                    target.angle = tar->FloatAttribute("angle", 0.0f);
                    target.compliance = tar->FloatAttribute("compliance", 0.0f);
                    target.jointName = uj->name;
                    targets.push_back(target);
                }
                joints.push_back(uj);
            }
            while ((joint = joint->NextSiblingElement("joint")));
        }

        // Load all cables
        XMLElement* cable = root->FirstChildElement("cable");
        while (cable)
        {
            URDFCable* cb = new URDFCable();

            cb->cyclic = cable->BoolAttribute("cyclic", false);
            cb->name = cable->Attribute("name");
            cb->stretchingCompliance = cable->FloatAttribute("stretchingCompliance", 0.0f);
            cb->compressionCompliance = cable->FloatAttribute("compressionCompliance", -1.0f);

            XMLElement* link = cable->FirstChildElement("link");
            while (link)
            {
                URDFCableLink lk;
                string type = link->Attribute("type");

                if (type == "fixed")
                {
                    lk.type = URDFCableLink::FIXED;
                    sscanf(link->Attribute("pos"), "%f %f %f", &lk.pos.x, &lk.pos.y, &lk.pos.z);
                }
                else if (type == "rolling")
                {
                    lk.type = URDFCableLink::ROLLING;
                    sscanf(link->Attribute("normal"), "%f %f %f", &lk.normal.x, &lk.normal.y, &lk.normal.z);
                    lk.d = link->FloatAttribute("d", 0.0f);
                    lk.extraLength = link->FloatAttribute("extraLength", 0.0f);
                }
                else if (type == "pinhole")
                {
                    lk.type = URDFCableLink::PINHOLE;
                    sscanf(link->Attribute("in"), "%f %f %f", &lk.in.x, &lk.in.y, &lk.in.z);
                    sscanf(link->Attribute("out"), "%f %f %f", &lk.out.x, &lk.out.y, &lk.out.z);
                }
                lk.body = linkH[link->Attribute("body")];
                cb->links.push_back(lk);
                link = link->NextSiblingElement("link");
            }
            cables.push_back(cb);
            cable = cable->NextSiblingElement("cable");
        }

        // Build joint hierarchy and convert everything to world space
        for (int i = 0; i < (int)links.size(); i++)
        {
            if (links[i]->parentJoint == nullptr)
            {
                // If no parent, start traversal
                traverseAndTransform(links[i], Transform());
            }
        }
        LLL;
    }

    void AddPhysicsEntities(Transform gt, int materialIndex, bool addVisualAttachments, float density = 1000.0f, float jointCompliance = 0.0f, float jointDamping = 10.f, float armature = 0.01f, 
                            float angularDamping = 20.7f, float maxAngularVelocity = 10.0f, bool addActiveJoints = true, float activeJointsCompliance = 1e-7f, float activeJointsDamping = 10.f, 
                            float thickness = 0.01f)
    {
        AddPhysicsEntities(gt, materialIndex, addVisualAttachments, false, density, jointCompliance, jointDamping, armature, angularDamping, maxAngularVelocity, 
                           addActiveJoints, activeJointsCompliance, activeJointsDamping, thickness);
    }

    void AddPhysicsEntities(Transform gt, int materialIndex, bool addVisualAttachments, bool flipVisualAttachments, float density = 1000.0f, float jointCompliance = 0.0f, float jointDamping = 1e1f, 
                            float armature = 0.01f, float angularDamping = 20.7f, float maxAngularVelocity = 10.0f, bool addActiveJoints = true, float activeJointsCompliance = 1e-7, float activeJointsDamping = 10.f, 
                            float thickness = 0.01f)
    {
        rigidNameMap.clear();
        urdfJointNameMap.clear();
        jointNameMap.clear();
        activeJointNameMap.clear();
        jointDOFNameMap.clear();
        vector<Transform> rigidTrans;
        int rigidBegin = g_buffers->rigidBodies.size();
        int shapeBegin = g_buffers->rigidShapes.size();

        for (int i = 0; i < (int)links.size(); i++)
        {
            URDFLink* l = links[i];
            if (l->mass >= 0.0f)
            {
                std::vector<NvFlexRigidShape> geometries;
                std::vector<float> densities;
                NvFlexRigidBody r;
                for (int g = 0; g < (int)l->colliders.size(); g++)
                {
                    // If mass == 0, will be ignored
                    Transform trans = Inverse(l->origin)*l->colliders[g].origin;
                    //printf("create %s\n", l->name.c_str());
                    if (l->colliders[g].type == URDFCollision::BOX)
                    {
                        NvFlexRigidShape shape;
                        NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), l->colliders[g].dim.x*0.5f, l->colliders[g].dim.y*0.5f, l->colliders[g].dim.z*0.5f, NvFlexMakeRigidPose(trans.p, trans.q));

                        shape.thickness = thickness;
                        geometries.push_back(shape);
                    }
                    else if (l->colliders[g].type == URDFCollision::SPHERE)
                    {
                        NvFlexRigidShape shape;
                        NvFlexMakeRigidSphereShape(&shape, g_buffers->rigidBodies.size(), l->colliders[g].dim.x, NvFlexMakeRigidPose(trans.p, trans.q));

                        shape.thickness = thickness;
                        geometries.push_back(shape);
                    }
                    else if (l->colliders[g].type == URDFCollision::CYLINDER)
                    {
                        NvFlexRigidShape shape;
                        if (replaceCylinderWithCapsule) 
                        {							
                            NvFlexMakeRigidCapsuleShape(&shape, g_buffers->rigidBodies.size(), l->colliders[g].dim.x, 0.5f*l->colliders[g].dim.y, NvFlexMakeRigidPose(trans.p, trans.q));
                        }
                        else 
                        {							
                            float r = l->colliders[g].dim.x;
                            float hlen = 0.5f*l->colliders[g].dim.y;
                            pair<float, float> rhp = make_pair(r, hlen);
                            if (cylinderCache.find(rhp) == cylinderCache.end()) 
                            {
                                Mesh* mesh = CreateCylinder(slicesPerCylinder, r, hlen, true);
                                NvFlexTriangleMeshId meshId = CreateTriangleMesh(mesh, 0);
                                cylinderCache[rhp] = meshId;
                            }
                            NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), cylinderCache[rhp], NvFlexMakeRigidPose(trans.p, trans.q), 1.0f, 1.0f, 1.0f);							
                        }

                        shape.thickness = thickness;
                        geometries.push_back(shape);
                    }
                    else if (l->colliders[g].type == URDFCollision::MESH)
                    {
                        for (size_t q = 0; q < l->colliders[g].geometries.size(); q++)
                        {
                            l->colliders[g].geometries[q].body = g_buffers->rigidBodies.size();

                            NvFlexRigidShape ts = l->colliders[g].geometries[q];
                            (Transform&)ts.pose = (trans * (Transform&)ts.pose);
                            geometries.push_back(ts);
                        }
                    }				
                }

                // TODO: support for per-shape densities
                densities.resize(geometries.size(), density);

                Transform trans = gt * l->origin;
                int shapeBegin = g_buffers->rigidShapes.size();
                int numBodyShapes = geometries.size();

                for (size_t s = 0; s < geometries.size(); s++)
                {
                    g_buffers->rigidShapes.push_back(geometries[s]);
                }

                NvFlexMakeRigidBody(g_flexLib, &r, trans.p, trans.q, &geometries[0], &densities[0], geometries.size());

                // assign inertia directly from the URDF (override calculated inertia)
                if (l->inertiaValid)
                {
                    r.mass = l->mass;
                    r.invMass = 1.0f / l->mass;
                    memcpy(&r.inertia[0], &l->inertia[0], sizeof(float) * 9);
                    Matrix33 m(r.inertia);
                    bool success;
                    Matrix33 im = Inverse(m, success);
                    memcpy(&r.invInertia[0], &im.cols[0].x, sizeof(float) * 9);
                }

                if (addVisualAttachments)
                {
                    for (int v = 0; v < (int)l->visuals.size(); v++)
                    {
                        if (l->visuals[v].meshes.size())
                        {
                            // add a render batch for visual attachments
                            for (size_t i = 0; i < l->visuals[v].meshes.size(); ++i)
                            {
                                const std::string& meshName = l->visuals[v].meshes[i];

                                Mesh* hostMesh = NULL;
                                RenderMesh* renderMesh = NULL;

                                auto it = meshCache.find(meshName);
                                if (it != meshCache.end())
                                {
                                    // found mesh in cache
                                    hostMesh = it->second.first;
                                    renderMesh = it->second.second;
                                }
                                else
                                {
                                    Mesh* mesh = ImportMesh(meshName.c_str());
                                    if (mesh)
                                    {
										#if URDF_FLATSHADE
											mesh->CalculateFaceNormals();
										#endif
                                        // flip for DAE files with z-up coordinate system
                                        if (flipVisualAttachments)
                                        {
                                            Matrix44 flip;
                                            flip(0, 0) = 1.0f;
                                            flip(2, 1) = 1.0f;
                                            flip(1, 2) = -1.0f;
                                            flip(3, 3) = 1.0f;
                                            mesh->Transform(flip);
                                        }

                                        mesh->Transform(ScaleMatrix(l->visuals[v].scale));

                                        // upload to GPU
                                        hostMesh = mesh;

                                        if (g_render == true)
                                        {
                                            renderMesh = CreateRenderMesh(mesh);
                                        }

                                        meshCache[meshName] = make_pair(hostMesh, renderMesh);
                                    }
                                }

                                if (hostMesh && renderMesh)
                                {
                                    // construct render batches
                                    for (size_t a=0; a < hostMesh->m_materialAssignments.size(); ++a)
                                    {
                                        MaterialAssignment assign = hostMesh->m_materialAssignments[a];

                                        RenderMaterial renderMaterial;
                                        renderMaterial.frontColor = hostMesh->m_materials[assign.material].Kd;
                                        renderMaterial.backColor = hostMesh->m_materials[assign.material].Kd;
                                        renderMaterial.specular = hostMesh->m_materials[assign.material].Ks.x;
                                        renderMaterial.roughness = SpecularExponentToRoughness(hostMesh->m_materials[assign.material].Ns);
                                        renderMaterial.metallic = hostMesh->m_materials[assign.material].metallic;

                                        // load texture
                                        char texturePath[2048];
                                        MakeRelativePath(meshName.c_str(), hostMesh->m_materials[assign.material].mapKd.c_str(), texturePath);

                                        renderMaterial.colorTex = CreateRenderTexture(texturePath);

                                        // construct render batch for this mesh/material combination
                                        RenderAttachment attach;
                                        attach.parent = g_buffers->rigidBodies.size();
                                        attach.material = renderMaterial;
                                        attach.mesh = renderMesh;
                                        attach.origin =  Inverse(l->origin)*l->visuals[v].origin;
                                        attach.startTri = hostMesh->m_materialAssignments[a].startTri;
                                        attach.endTri = hostMesh->m_materialAssignments[a].endTri;

                                        g_renderAttachments.push_back(attach);
                                    }

                                    // if no material assignment then create a default one based on the link
                                    if (hostMesh->m_materialAssignments.empty())
                                    {
										
                                        RenderAttachment attach;
                                        attach.parent = g_buffers->rigidBodies.size();                                       
                                        attach.material.frontColor = Vec3(SrgbToLinear(Colour(l->visuals[v].material.color)));
                                        attach.material.backColor = Vec3(SrgbToLinear(Colour(l->visuals[v].material.color)));
                                        attach.material.specular = 0.5f;
                                        attach.material.metallic = 0.0f;
                                        attach.mesh = renderMesh;
                                        attach.origin =  Inverse(l->origin)*l->visuals[v].origin;
                                        attach.startTri = 0;
                                        attach.endTri = 0;

                                        g_renderAttachments.push_back(attach);
                                    }
                                }
                            }
                        }
                    }
                }

                // increase body mass and inertia
                (Matrix33&)r.inertia += armature * Matrix33::Identity();

                bool succ;
                (Matrix33&)r.invInertia = Inverse(Matrix33(r.inertia), succ);

                rigidTrans.push_back(trans);
                rigidMap[l] = g_buffers->rigidBodies.size();
                rigidNameMap[l->name] = g_buffers->rigidBodies.size();
                //debugLabelPos.push_back(trans.p);
                //debugLabelString.push_back(l->name);
                rigidShapeNameMap[l->name] = make_pair(shapeBegin, numBodyShapes);
                l->flexBodyIndex = (int)g_buffers->rigidBodies.size();

                // global damping
                r.angularDamping = angularDamping;
                r.maxAngularVelocity = maxAngularVelocity;
#if URDF_VERBOSE
                printf("Adding Link: %s Index: %d\n", l->name.c_str(), g_buffers->rigidBodies.size());
#endif
                g_buffers->rigidBodies.push_back(r);
            }
        }

        // assign render material to all new shapes
        for (int i = shapeBegin; i < (int)g_buffers->rigidShapes.size(); ++i)
        {
            g_buffers->rigidShapes[i].user = UnionCast<void*>(materialIndex);
        }

        for (int i = 0; i < (int)joints.size(); i++)
        {
            URDFJoint* j = joints[i];
            Transform origin = j->origin;
            Vec3 rotAxis = -Cross(j->axis, Vec3(1.0f, 0.0f, 0.0f));
            if (Dot(rotAxis, rotAxis) < 1e-5)
            {
                rotAxis = Vec3(0.0f, 1.0f, 0.0f);
            }
            else
            {
                rotAxis /= sqrtf(Dot(rotAxis, rotAxis));
            }

            Quat alignQ = QuatFromAxisAngle(rotAxis, acos(j->axis.x));

            origin.q = j->origin.q * alignQ;

            origin = gt * origin;
            int parent = -1;
            if (rigidMap.find(j->parent) != rigidMap.end())
            {
                parent = rigidMap[j->parent];
            }

            int child = -1;
            if (rigidMap.find(j->child) != rigidMap.end())
            {
                child = rigidMap[j->child];
            }

            if (child == -1)
            {
                continue;    // Ignore joint if child == -1
            }

            Transform ptran = (parent >= 0) ? rigidTrans[parent - rigidBegin] : Transform();
            Transform ctran = (child >= 0) ? rigidTrans[child - rigidBegin] : gt;

            Transform xppose = Inverse(ptran) * origin;
            Transform xcpose = Inverse(ctran) * origin;

            NvFlexRigidPose ppose = NvFlexMakeRigidPose(xppose.p, xppose.q);
            NvFlexRigidPose cpose = NvFlexMakeRigidPose(xcpose.p, xcpose.q);
            NvFlexRigidJoint joint;
            int dof = eNvFlexRigidJointAxisTwist;

            if (j->type == URDFJoint::CONTINUOUS)
            {
                NvFlexMakeHingeJoint(&joint, parent, child, ppose, cpose, eNvFlexRigidJointAxisTwist);
            }
            else if (j->type == URDFJoint::REVOLUTE)
            {
                NvFlexMakeHingeJoint(&joint, parent, child, ppose, cpose, eNvFlexRigidJointAxisTwist, j->lower, j->upper);
                joint.compliance[eNvFlexRigidJointAxisTwist] = jointCompliance; // 10^6 N/m
                joint.damping[eNvFlexRigidJointAxisTwist] = jointDamping;   // 5*10^5 N/m/s
            }
            else if (j->type == URDFJoint::PRISMATIC)
            {
                NvFlexMakePrismaticJoint(&joint, parent, child, ppose, cpose, eNvFlexRigidJointAxisX, j->lower, j->upper);
                joint.compliance[eNvFlexRigidJointAxisX] = jointCompliance;
                joint.damping[eNvFlexRigidJointAxisX] = jointDamping;
                dof = eNvFlexRigidJointAxisX;
            }
            else
            {
                NvFlexMakeFixedJoint(&joint, parent, child, ppose, cpose);
                dof = -1;
            }

            urdfJointNameMap[j->name] = i;
            jointNameMap[j->name] = g_buffers->rigidJoints.size();
            jointDOFNameMap[j->name] = dof;
            j->flexJointIndex = (int)g_buffers->rigidJoints.size();
            g_buffers->rigidJoints.push_back(joint);

            if (addActiveJoints)
            {
                for (int k = 0; k < 6; ++k)
                {
                    joint.modes[k] = eNvFlexRigidJointModeFree;
                }

                // Now actuator joint, set to the middle of the limit
                if (j->type == URDFJoint::CONTINUOUS)
                {
                    NvFlexMakeHingeJoint(&joint, parent, child, ppose, cpose, eNvFlexRigidJointAxisTwist);
                    joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
                    joint.compliance[eNvFlexRigidJointAxisTwist] = activeJointsCompliance;
                    joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
                    joint.damping[eNvFlexRigidJointAxisTwist] = activeJointsDamping;
                }
                else if (j->type == URDFJoint::REVOLUTE)
                {
                    NvFlexMakeHingeJoint(&joint, parent, child, ppose, cpose, eNvFlexRigidJointAxisTwist);
                    joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
                    joint.compliance[eNvFlexRigidJointAxisTwist] = activeJointsCompliance;
                    joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
                    joint.damping[eNvFlexRigidJointAxisTwist] = activeJointsDamping;

                }
                else if (j->type == URDFJoint::PRISMATIC)
                {
                    joint.targets[eNvFlexRigidJointAxisX] = 0.0f;
                    joint.compliance[eNvFlexRigidJointAxisX] = activeJointsCompliance;
                    joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
                    joint.damping[eNvFlexRigidJointAxisX] = activeJointsDamping;
                }
                if (dof != -1)
                {
                    activeJointNameMap[j->name] = g_buffers->rigidJoints.size();
                    g_buffers->rigidJoints.push_back(joint);
                }

            }

            if (j->mimic != "")
            {
                // Mimic

                for (int k = 0; k < 6; ++k)
                {
                    joint.modes[k] = eNvFlexRigidJointModeFree;
                }

                if ((j->type == URDFJoint::CONTINUOUS) || (j->type == URDFJoint::REVOLUTE))
                {
                    cout << "Mimic revolute / continuous joint is not ye supported!" << endl;
                    exit(0);
                    joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeMimic;
                    joint.compliance[eNvFlexRigidJointAxisTwist] = 0.0f;
                }
                else if (j->type == URDFJoint::PRISMATIC)
                {
                    joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeMimic;
                    joint.compliance[eNvFlexRigidJointAxisX] = 0.0f;
                }

                joint.mimicIndex = jointNameMap[j->mimic];
                (Vec3&)joint.mimicScale = Vec3(1.0f, 0.0f, 0.0f);
                (Vec3&)joint.mimicOffset = Vec3(0.0f);
                g_buffers->rigidJoints.push_back(joint);
            }
        }
        for (int i = 0; i < cables.size(); i++)
        {
            URDFCable& cable = *cables[i];
            vector<NvFlexCableLink> cbl;
            vector<const NvFlexRigidBody*> bodies;

            cbl.resize(cable.links.size());
            int linkStartIndex = g_buffers->cableLinks.size();
            for (int j = 0; j < (int)cable.links.size(); j++)
            {
                NvFlexCurveId cid = rigidCurveMap[cable.links[j].body->name + string("_") + to_string(i) + string(":") + to_string(j)];
                bool noworld = !(cable.links[j].body->name == "world");

                if (cable.links[j].type == URDFCableLink::FIXED)
                {
                    NvFlexInitFixedCableLink(g_flexLib, &cbl[j], linkStartIndex + j, noworld ? &g_buffers->rigidBodies[rigidNameMap[cable.links[j].body->name]] : 0,
                                             noworld ? &g_buffers->rigidShapes[rigidShapeNameMap[cable.links[j].body->name].first] : 0, rigidShapeNameMap[cable.links[j].body->name].second, TransformPoint(gt, cable.links[j].pos));
                }
                else if (cable.links[j].type == URDFCableLink::ROLLING)
                {
                    Vec3 newN = Rotate(gt.q, cable.links[j].normal);
                    NvFlexInitRollingCableLink(g_flexLib, &cbl[j], linkStartIndex + j, noworld ? &g_buffers->rigidBodies[rigidNameMap[cable.links[j].body->name]] : 0,
                                               noworld ? &g_buffers->rigidShapes[rigidShapeNameMap[cable.links[j].body->name].first] : 0, rigidShapeNameMap[cable.links[j].body->name].second, Vec4(newN.x, newN.y, newN.z, cable.links[j].d - Dot(gt.p, newN)), cable.links[j].extraLength, cid);
                }
                else if (cable.links[j].type == URDFCableLink::PINHOLE)
                {
                    NvFlexInitPinholeCableLink(g_flexLib, &cbl[j], linkStartIndex + j, noworld ? &g_buffers->rigidBodies[rigidNameMap[cable.links[j].body->name]] : 0,
                                               noworld ? &g_buffers->rigidShapes[rigidShapeNameMap[cable.links[j].body->name].first] : 0, rigidShapeNameMap[cable.links[j].body->name].second, TransformPoint(gt, cable.links[j].in), TransformPoint(gt, cable.links[j].out));
                }
                bodies.push_back(noworld ? &g_buffers->rigidBodies[rigidNameMap[cable.links[j].body->name]] : 0);
                cid = cbl[j].profileVerts;
                rigidCurveMap[cable.links[j].body->name + string("_") + to_string(i) + string(":") + to_string(j)] = cid; // Store curve
            }
            NvFlexMakeCable(g_flexLib, &cbl[0], &bodies[0], cbl.size(), cable.cyclic, cable.stretchingCompliance, 0.0f, cable.compressionCompliance, 0.0f);
            for (int j = 0; j < (int)cable.links.size(); j++)
            {
                g_buffers->cableLinks.push_back(cbl[j]);
            }
        }
        /*
           for (int i = 0; i < cables.size(); i++)
           {
           URDFCable& cable = *cables[i];
           for (int j = 0; j < (int)cable.links.size(); j++)
           {
           if (cable.links[j].type == URDFCableLink::FIXED)
           {
           debugPoints.push_back(TransformPoint(gt, cable.links[j].pos));
           debugPointsCols.push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
           }
           else if (cable.links[j].type == URDFCableLink::ROLLING)
           {
           Vec3 n = Rotate(gt.q, cable.links[j].normal);
           float d = cable.links[j].d - Dot(gt.p, n);
           debugPlanes.push_back(Vec4(n.x, n.y, n.z, d));
           debugPlanesString.push_back(cable.links[j].body->name);
           }
           else if (cable.links[j].type == URDFCableLink::PINHOLE)
           {
           debugPoints.push_back(TransformPoint(gt, cable.links[j].in));
           debugPoints.push_back(TransformPoint(gt, cable.links[j].out));

           debugPointsCols.push_back(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
           debugPointsCols.push_back(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
           }
           }
           }
           */
    }

    void setReducedPose(map<string, Transform>& lposes)
    {
        for (int i = 0; i < (int)links.size(); i++)
        {
            if (links[i]->parentJoint == nullptr)
            {
                // If no parent, start traversal
                setReducedPose(links[i], lposes);
            }
        }
    }

    void setReducedPose(URDFJoint* joint, map<string, Transform>& lposes)
    {
        if (joint->child)
        {
            setReducedPose(joint->child, lposes);
        }

    }
    void setReducedPose(URDFLink* link, map<string, Transform>& lposes)
    {
        URDFJoint* j = link->parentJoint;
        //cout << "Set joint" << endl;
        if (j)
        {
            NvFlexRigidJoint& rj = g_buffers->rigidJoints[j->flexJointIndex];
            Transform pt;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[rj.body0], (NvFlexRigidPose*)&pt);
            Transform worldJt;
            worldJt = pt * (Transform&)rj.pose0;
            if (lposes.find(link->name) != lposes.end())
            {
                worldJt = worldJt * lposes[link->name];
            }
            Transform ct = worldJt * Inverse((Transform&)rj.pose1);
            NvFlexSetRigidPose(&g_buffers->rigidBodies[rj.body1], (NvFlexRigidPose*)&ct);
        }

        // Not sure!
        for (int i = 0; i < (int)link->childJoints.size(); i++)
        {
            setReducedPose(link->childJoints[i], lposes);
        }
    }

};
