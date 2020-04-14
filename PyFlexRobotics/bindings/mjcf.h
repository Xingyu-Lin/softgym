#pragma once

#include "../external/tinyxml2/tinyxml2.h"
using namespace tinyxml2;

class MJCFJoint
{
public:
    enum Type { HINGE, SLIDE };
    float armature;
    float damping;
    bool limited;
    Vec3 axis;
    float ref;
    string name;

    Vec3 pos;
    Vec2 range;
    float stiffness;
    Type type;
	float initVal;

    MJCFJoint()
    {
        armature = 0.0f;
        damping = 0.0f;
        limited = false;
        axis = Vec3(1.0f, 0.0f, 0.0f);
        name = "";

        pos = Vec3(0.0f, 0.0f, 0.0f);
        range = Vec2(0.0f, 0.0f);
        stiffness = 0.0f;
        type = HINGE;

        ref = 0.0f;
		initVal = 0.0f;
    }
};

class MJCFGeom
{
public:
    enum Type { CAPSULE, SPHERE, ELLIPSOID, CYLINDER, BOX, MESH};

    float density;
    int conaffinity;
    int condim;
    int contype;
    float margin;

    Vec3 friction; // Sliding Torsion Rolling
    string material;
    Vec4 rgba;
    Vec3 solimp;
    Vec2 solref;
    Vec3 from;
    Vec3 to;
    Vec3 size;
    string name;
    Vec3 pos;
    Type type;
    Quat quat;
    Vec4 axisangle;
	Vec3 zaxis;

	std::string mesh;

    bool hasFromTo;
    bool hasAxisAngle;
	bool hasZAxis;
    bool hasQuat;

    MJCFGeom()
    {
        conaffinity = 1;
        condim = 3;
        contype = 1;
        margin = 0.0f;
        friction = Vec3(1.0f, 0.005f, 0.0001f);
        material = "";
        rgba = Vec4(0.9f, 0.5f, 0.2f, 1.0f);
        solimp = Vec3(0.9f, 0.95f, 0.001f);
        solref = Vec2(0.02f, 1.0f);
        from = Vec3(0.0f, 0.0f, 0.0f);
        to = Vec3(0.0f, 0.0f, 0.0f);
        size = Vec3(0.0f, 0.0f, 0.0f);
        name = "";
        pos = Vec3(0.0f, 0.0f, 0.0f);
        type = SPHERE;
        density = 1000.0f;

        axisangle = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        quat = Quat();
        hasFromTo = false;
        hasAxisAngle = false;
		hasZAxis = false;
        hasQuat = false;
    }
};

class MJCFMotor
{
public:
    bool ctrllimited;
    Vec2 ctrlrange;
    float gear;
    string joint;
    string name;

    MJCFMotor()
    {
        ctrllimited = false;
        ctrlrange = Vec2(-1.0f, 1.0f);
        gear = 0.0f;
        name = "";
        joint = "";
    }
};

class MJCFBody
{
public:
    string name;
    Vec3 pos;
    Quat quat;
    vector<MJCFGeom*> geoms;
    vector<MJCFJoint*> joints;
    vector<MJCFBody*> bodies;

    MJCFBody()
    {
        name = "";
        pos = Vec3(0.0f, 0.0f, 0.0f);
        quat = Quat();
        geoms.clear();
        joints.clear();
        bodies.clear();
    }

    ~MJCFBody()
    {
        for (int i = 0; i < (int)geoms.size(); i++)
        {
            delete geoms[i];
        }

        for (int i = 0; i < (int)joints.size(); i++)
        {
            delete joints[i];
        }

        for (int i = 0; i < (int)bodies.size(); i++)
        {
            delete bodies[i];
        }
    }
};

class MJCFCompiler
{
public:
    bool angleInRad;
    bool inertiafromgeom;
    bool coordinateInLocal;
	string eulerseq;
	string meshDir;

    MJCFCompiler()
    {
		eulerseq = "xyz";
        angleInRad = false;
        inertiafromgeom = true;
        coordinateInLocal = true;		
    }
};

class MJCFClass
{
public:
	MJCFJoint djoint;
	MJCFGeom dgeom;
	MJCFMotor dmotor;
	string name;
};

class Vec3Comp
{
public:
	bool operator () (const Vec3& a, const Vec3& b) const
	{
		if (a.x < b.x) return true;
		if (a.x > b.x) return false;
		if (a.y < b.y) return true;
		if (a.y > b.y) return false;
		return a.z < b.z;
	}
};

class Vec2Comp
{
public:
	bool operator () (const Vec2& a, const Vec2& b) const
	{
		if (a.x < b.x) return true;
		if (a.x > b.x) return false;
		return a.y < b.y;
	}
};

class MJCFImporter
{
public:
	string defaultClassName;
	map<string, MJCFClass> classes;

    MJCFCompiler compiler;
    vector<MJCFBody*> bodies;
    vector<MJCFMotor*> motors;

    map<string, int> bmap;
    map<string, pair<int, NvFlexRigidJointAxis>> jmap;
    map<string, pair<int, NvFlexRigidJointAxis>> d6jmap;
	map<string, int> controlMap; // jointName -> control index

    map<int, Transform> rigidTrans;
    map<int, pair<int, int> > myShapes;
    map<string, pair<int, Transform> > geoBodyPose; // Local pose of geo and body with respect to body

	map<string, NvFlexTriangleMeshId> assets;

    int d6jointCounter;

    bool createActiveJoints;
	bool createBodyForFixedJoint;
    vector<pair<int, NvFlexRigidJointAxis> > activeJoints; // Joint index, dof index
    map<string, int> activeJointsNameMap; // Map name to index in activeJoints

	int firstBody;

    ~MJCFImporter()
    {
        for (int i = 0; i < (int)bodies.size(); i++)
        {
            delete bodies[i];
        }
        for (int i = 0; i < (int)motors.size(); i++)
        {
            delete motors[i];
        }
    }

    void getIfExist(XMLElement* e, const char* aname, bool& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            string s = st;
            if (s == "true")
            {
                p = true;
            }
            if (s == "1")
            {
                p = true;
            }
            if (s == "false")
            {
                p = false;
            }
            if (s == "0")
            {
                p = false;
            }
        }
    }
    void getIfExist(XMLElement* e, const char* aname, int& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%d", &p);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, float& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f", &p);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, string& s)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            s = st;
        }
    }
    void getIfExist(XMLElement* e, const char* aname, Vec2& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f %f", &p.x, &p.y);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, Vec3& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f %f %f", &p.x, &p.y, &p.z);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, Vec3& from, Vec3& to)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f %f %f %f %f %f", &from.x, &from.y, &from.z, &to.x, &to.y, &to.z);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, Vec4& p)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f %f %f %f", &p.x, &p.y, &p.z, &p.w);
        }
    }
    void getIfExist(XMLElement* e, const char* aname, Quat& q)
    {
        const char* st = e->Attribute(aname);
        if (st)
        {
            sscanf(st, "%f %f %f %f", &q.w, &q.x, &q.y, &q.z);
			q = Normalize(q);
        }
    }

	void getEulerIfExist(XMLElement* e, const char* aname, Quat& q)
	{
		const char* st = e->Attribute(aname);
		if (st)
		{
			// Euler
			if (compiler.eulerseq != "xyz") {
				cout << "Only support xyz Euler seq" << endl;
				exit(0);
			}
			float a, b, c;
			sscanf(st, "%f %f %f", &a, &b, &c);
			if (!compiler.angleInRad) {
				a = kPi * a / 180.0f;
				b = kPi * b / 180.0f;
				c = kPi * c / 180.0f;
			}
			q = rpy2quat(a, b, c);
			
		}
	}

    string GetAttr(XMLElement* c, const char* name)
    {
        if (c->Attribute(name))
        {
            return string(c->Attribute(name));
        }
        else
        {
            return "";
        }
    }

    void LoadCompiler(XMLElement* c)
    {
        if (c)
        {
            string s;
			
			if ((s = GetAttr(c, "eulerseq")) != "")
			{
				if (s != "xyz") {
					cout << "Euler sequence other than xyz is not supported now..." << endl;
					exit(0);
				}
			}

            if ((s = GetAttr(c, "angle")) != "")
            {
                compiler.angleInRad = (s == "radian");
            }

            if ((s = GetAttr(c, "inertiafromgeom")) != "")
            {
                compiler.inertiafromgeom = (s == "true");
            }

            if ((s = GetAttr(c, "coordinate")) != "")
            {
                compiler.coordinateInLocal = (s == "local");
                if (!compiler.coordinateInLocal)
                {
                    cout << "Don't know how to handle global coordinate yet!" << endl;
                    exit(0);
                }
            }

			getIfExist(c, "meshdir", compiler.meshDir);

			// load assets

        }
    }

    void LoadGeom(XMLElement* g, MJCFGeom& geom, string className)
    {
        if (!g)
        {
            return;
        }
		if (g->Attribute("class")) className = g->Attribute("class");
		geom = classes[className].dgeom;

        getIfExist(g, "conaffinity", geom.conaffinity);
        getIfExist(g, "condim", geom.condim);
        getIfExist(g, "contype", geom.contype);
        getIfExist(g, "margin", geom.margin);
        getIfExist(g, "friction", geom.friction);
        getIfExist(g, "material", geom.material);
        getIfExist(g, "rgba", geom.rgba);
        getIfExist(g, "solimp", geom.solimp);
        getIfExist(g, "solref", geom.solref);
        getIfExist(g, "fromto", geom.from, geom.to);
        getIfExist(g, "axisangle", geom.axisangle);
		getIfExist(g, "zaxis", geom.zaxis);
        getIfExist(g, "size", geom.size);
        getIfExist(g, "name", geom.name);
        getIfExist(g, "pos", geom.pos);
		getEulerIfExist(g, "euler", geom.quat);
		getIfExist(g, "quat", geom.quat);
        getIfExist(g, "density", geom.density);
		getIfExist(g, "mesh", geom.mesh);

        if (g->Attribute("fromto"))
        {
            geom.hasFromTo = true;
        }
        if (g->Attribute("axisangle"))
        {
            geom.hasAxisAngle = true;
        }
		if (g->Attribute("zaxis"))
		{
			geom.hasZAxis = true;
		}
        if (g->Attribute("quat"))
        {
            geom.hasQuat = true;
        }

        string type = "";
        getIfExist(g, "type", type);
        if (type == "capsule")
        {
            geom.type = MJCFGeom::CAPSULE;
        }
        else if (type == "sphere")
        {
            geom.type = MJCFGeom::SPHERE;
        }
		else if (type == "ellipsoid")
		{
			cout << "Ellipsoid is not natively supported, tesellated mesh will be used" << endl;
			geom.type = MJCFGeom::ELLIPSOID;
		}
		else if (type == "cylinder")
		{
			cout << "Cylinder is not natively supported, tesellated mesh will be used" << endl;
			geom.type = MJCFGeom::CYLINDER;
		}
        else if (type == "box")
        {
            geom.type = MJCFGeom::BOX;
        }
		else if (type == "mesh")
		{
			geom.type = MJCFGeom::MESH;
		}
        else if (type != "")
        {
			cout << "Geom type " << type << " not yet supported!" << endl;
        }

        if (geom.hasAxisAngle)
        {
            // Convert to quat
            if (!compiler.angleInRad)
            {
                geom.axisangle.w = kPi * geom.axisangle.w / 180.0f;
            }
            geom.quat = QuatFromAxisAngle(Vec3(geom.axisangle.x, geom.axisangle.y, geom.axisangle.z), geom.axisangle.w);
        }

		if (geom.hasZAxis)
		{
			// Convert to quat
			geom.quat = Quat(geom.zaxis);
		}
    }

    void LoadJoint(XMLElement* g, MJCFJoint& joint, string className)
    {
        if (!g)
        {
            return;
        }
		if (g->Attribute("class")) className = g->Attribute("class");
		joint = classes[className].djoint;

        getIfExist(g, "ref", joint.ref);
        getIfExist(g, "armature", joint.armature);
        getIfExist(g, "damping", joint.damping);
        getIfExist(g, "limited", joint.limited);
        getIfExist(g, "axis", joint.axis);
        getIfExist(g, "name", joint.name);
        getIfExist(g, "pos", joint.pos);
        getIfExist(g, "range", joint.range);
        if (!compiler.angleInRad)
        {
			//cout << "Angle in deg" << endl;
            joint.range.x = kPi * joint.range.x / 180.0f;
            joint.range.y = kPi * joint.range.y / 180.0f;
        }
        getIfExist(g, "stiffness", joint.stiffness);
        string type = "";

        getIfExist(g, "type", type);
        if (type == "hinge")
        {
            joint.type = MJCFJoint::HINGE;
        }
        else if (type == "slide")
        {
            joint.type = MJCFJoint::SLIDE;
        }
        else if (type != "")
        {
            cout << "Joint type " << type << " not yet supported!" << endl;
        }
        joint.axis = Normalize(joint.axis);
    }

    void LoadMotor(XMLElement* g, MJCFMotor& motor, string className)
    {
        if (!g)
        {
            return;
        }

		if (g->Attribute("class"))
		{
			className = g->Attribute("class");
		}
		motor = classes[className].dmotor;
        getIfExist(g, "ctrllimited", motor.ctrllimited);
        getIfExist(g, "ctrlrange", motor.ctrlrange);
        getIfExist(g, "gear", motor.gear);
        getIfExist(g, "joint", motor.joint);
        getIfExist(g, "name", motor.name);
    }

    void LoadDefault(XMLElement* e, string className, MJCFClass& cl)
    {
        LoadJoint(e->FirstChildElement("joint"), cl.djoint, className);
        LoadMotor(e->FirstChildElement("motor"), cl.dmotor, className);
        LoadGeom(e->FirstChildElement("geom"), cl.dgeom, className);

		XMLElement* d = e->FirstChildElement("default");
		while (d)
		{
			// While there is child default
			// Must have name 
			if (!d->Attribute("class"))
			{
				cout << "Non-top level class must have name" << endl;
				exit(0);
			}

			string name = d->Attribute("class");
			classes[name] = cl; // Copy from this class
			LoadDefault(d, name, classes[name]); // Recursively load it
			d = d->NextSiblingElement("default");
		}
    }

    void LoadBody(XMLElement* g, MJCFBody& body, string className)
    {
        if (!g)
        {
            return;
        }

		if (g->Attribute("childclass"))
		{
			className = g->Attribute("childclass");
		}
        getIfExist(g, "name", body.name);
        getIfExist(g, "pos", body.pos);
		getEulerIfExist(g, "euler", body.quat);
        getIfExist(g, "quat", body.quat);

        // Load geoms
        XMLElement* c = g->FirstChildElement("geom");
        while (c)
        {
            body.geoms.push_back(new MJCFGeom());
            LoadGeom(c, *body.geoms.back(), className);
            c = c->NextSiblingElement("geom");
        }
        // Load joints
        c = g->FirstChildElement("joint");
        while (c)
        {
            body.joints.push_back(new MJCFJoint());
            LoadJoint(c, *body.joints.back(), className);
            c = c->NextSiblingElement("joint");
        }
        // Load child bodies
        c = g->FirstChildElement("body");
        while (c)
        {
            body.bodies.push_back(new MJCFBody());
            LoadBody(c, *body.bodies.back(), className);
            c = c->NextSiblingElement("body");
        }
    }

	void createPhysicsBodyAndJoint(MJCFBody* body, Transform trans, int parent, float maxAngularVelocity, bool collideEachOther = true)
	{
		// Create body (compound of geoms)
		std::vector<NvFlexRigidShape> geometries;
		NvFlexRigidBody me;

		std::vector<float> densities;
		Vec3 friction = body->geoms[0]->friction;
		int shapeBegin = g_buffers->rigidShapes.size();
		int myIndex = parent;

		Transform myTrans = trans * Transform(body->pos, body->quat);
		if ( (!createBodyForFixedJoint) && ((body->joints.size() == 0) && (parent != -1)) )
		{			
			cout << "Body with no joint will have no geometry for now, to avoid instability of fixed joint!" << endl;
		} 
		else 
		{
			for (int i = 0; i < (int)body->geoms.size(); i++)
			{
				densities.push_back(body->geoms[i]->density);

				if (Length(body->geoms[i]->friction - friction) > 1e-5f)
				{
					cout << "Unequal friction not supported now..." << endl;
				}

				NvFlexRigidShape shape;
				memset(&shape, 0, sizeof(shape));

				shape.material.friction = friction.x;
				shape.material.torsionFriction = friction.y;
				shape.material.rollingFriction = friction.z;
				shape.material.restitution = 0.0f;
				shape.material.compliance = 0.0f;

				const Vec3 color = Vec3(body->geoms[i]->rgba.x, body->geoms[i]->rgba.y, body->geoms[i]->rgba.z);
				shape.user = UnionCast<void*>(AddRenderMaterial(color, 0.3f, 0.4f));

				if (body->geoms[i]->type == MJCFGeom::SPHERE)
				{
					shape.body = g_buffers->rigidBodies.size();
					shape.geo.sphere.radius = body->geoms[i]->size.x;
					shape.geoType = eNvFlexShapeSphere;
					shape.pose = NvFlexMakeRigidPose(body->geoms[i]->pos, body->geoms[i]->quat);
				}
				else if (body->geoms[i]->type == MJCFGeom::ELLIPSOID)
				{
					float mPerSeg = 0.02f;
					float a = body->geoms[i]->size.x;
					float b = body->geoms[i]->size.z;
					// Perimeter of equator approximated with Ramanujan's O(h^5) approximation
					float h = powf(a - b, 2.0f) / powf(a + b, 2.0f);
					float pXZ = kPi * (a + b) * (1.0f + 3.0f * h / (10.0f + sqrtf(4.0f - 3.0f * h)));

					b = body->geoms[i]->size.y;
					h = powf(a - b, 2.0f) / powf(a + b, 2.0f);
					float pXY = kPi * (a + b) * (1.0f + 3.0f*h / (10.0f + sqrtf(4.0f - 3.0f * h)));

					Mesh* mesh = CreateEllipsoid((int)(pXY * 0.5f / mPerSeg), (int)(pXZ / mPerSeg), body->geoms[i]->size);
					NvFlexTriangleMeshId meshId = CreateTriangleMesh(mesh, 0.0f);
					NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), meshId, NvFlexMakeRigidPose(body->geoms[i]->pos, body->geoms[i]->quat), 1.0f, 1.0f, 1.0f);
				}
				else if (body->geoms[i]->type == MJCFGeom::BOX)
				{
					shape.body = g_buffers->rigidBodies.size();
					shape.geo.box.halfExtents[0] = body->geoms[i]->size.x;
					shape.geo.box.halfExtents[1] = body->geoms[i]->size.y;
					shape.geo.box.halfExtents[2] = body->geoms[i]->size.z;
					shape.geoType = eNvFlexShapeBox;
					shape.pose = NvFlexMakeRigidPose(body->geoms[i]->pos, body->geoms[i]->quat);
				}
				else if (body->geoms[i]->type == MJCFGeom::CAPSULE)
				{
					Vec3 cen;
					Quat q;
					float hlen;

					if (body->geoms[i]->hasFromTo)
					{
						cen = 0.5f * (body->geoms[i]->from + body->geoms[i]->to);
						Vec3 dif = body->geoms[i]->to - body->geoms[i]->from;

						hlen = 0.5f * Length(dif);
						dif = Normalize(dif);

						Vec3 rotVec = Cross(Vec3(1.0f, 0.0f, 0.0f), dif);
						if (Length(rotVec) < 1e-5)
						{
							rotVec = Vec3(0.0f, 1.0f, 0.0f);
						}
						else
						{
							rotVec = Normalize(rotVec);
						}

						float angle = acos(dif.x);
						q = QuatFromAxisAngle(rotVec, angle);
					}
					else
					{
						cen = body->geoms[i]->pos;
						q = body->geoms[i]->quat * QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f);
						hlen = body->geoms[i]->size.y;
					}
					shape.body = g_buffers->rigidBodies.size();
					shape.geo.capsule.radius = body->geoms[i]->size.x;
					shape.geo.capsule.halfHeight = hlen;
					shape.geoType = eNvFlexShapeCapsule;
					shape.pose = NvFlexMakeRigidPose(cen, q);
				}
				else if (body->geoms[i]->type == MJCFGeom::CYLINDER)
				{
					Vec3 cen;
					Quat q;
					float hlen;

					if (body->geoms[i]->hasFromTo)
					{
						cen = 0.5f * (body->geoms[i]->from + body->geoms[i]->to);
						Vec3 dif = body->geoms[i]->to - body->geoms[i]->from;

						hlen = 0.5f * Length(dif);
						dif = Normalize(dif);

						Vec3 rotVec = Cross(Vec3(1.0f, 0.0f, 0.0f), dif);
						if (Length(rotVec) < 1e-5)
						{
							rotVec = Vec3(0.0f, 1.0f, 0.0f);
						}
						else
						{
							rotVec = Normalize(rotVec);
						}

						float angle = acos(dif.x);
						q = QuatFromAxisAngle(rotVec, angle);
					}
					else
					{
						cen = body->geoms[i]->pos;
						q = body->geoms[i]->quat * QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f);
						hlen = body->geoms[i]->size.y;
					}

					float mPerSeg = 0.02f;
					float r = body->geoms[i]->size.x;
					Mesh* mesh = CreateCylinder((int)(2.0f*kPi*r / mPerSeg), r, hlen, true);
					NvFlexTriangleMeshId meshId = CreateTriangleMesh(mesh, 0.0f);
					NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), meshId, NvFlexMakeRigidPose(body->geoms[i]->pos, body->geoms[i]->quat*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi * 0.5f)), 1.0f, 1.0f, 1.0f);

					/*
					shape.body = g_buffers->rigidBodies.size();
					shape.geo.capsule.radius = body->geoms[i]->size.x;
					shape.geo.capsule.halfHeight = hlen;
					shape.geoType = eNvFlexShapeCapsule;
					shape.pose = NvFlexMakeRigidPose(cen, q);
					*/
				}
				else if (body->geoms[i]->type == MJCFGeom::MESH)
				{				
					NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), assets[body->geoms[i]->mesh], NvFlexMakeRigidPose(body->geoms[i]->pos, body->geoms[i]->quat), 1.0f, 1.0f, 1.0f);
				}

				// Set collision filter
				if (collideEachOther && body->geoms[i]->conaffinity)
				{
					shape.filter = 0;
				}
				else
				{
					shape.filter = 1;
				}

				geometries.push_back(shape);

				geoBodyPose[body->geoms[i]->name] = make_pair(g_buffers->rigidBodies.size(), (Transform&)shape.pose);
				g_buffers->rigidShapes.push_back(shape);
			}
	
			NvFlexMakeRigidBody(g_flexLib, &me, myTrans.p, myTrans.q, &geometries[0], &densities[0], geometries.size());

			geoBodyPose[body->name] = make_pair(parent, Transform(body->pos, body->quat));

			// clamp max angular  velocity
			me.maxAngularVelocity = maxAngularVelocity;
			g_buffers->rigidBodies.push_back(me);

			myIndex = g_buffers->rigidBodies.size() - 1;
			bmap[body->name] = myIndex;
			rigidTrans[myIndex] = myTrans;
			myShapes[myIndex] = make_pair(shapeBegin, (int)body->geoms.size());

			// Create joint linked to parent
			if (parent != -1)
			{
				Transform origin; // Joint transform
				bool sameOrigin = true;
				bool allHinge = true;
				if (body->joints.size() > 0)
				{
					origin.p = body->joints[0]->pos; // Origin at last joint (deepest)

					for (int i = 1; i < (int)body->joints.size(); i++)
					{
						if (LengthSq(origin.p - body->joints[i]->pos) > 1e-5)
						{
							sameOrigin = false;
							//cout << "Don't know how to handle joints with unequal pos....." << endl;
							//exit(0);
						}
					}
				}
				else
				{
					origin.p = Vec3(0.0f, 0.0f, 0.0f);
				}

				// Build Local Frame
				if (body->joints.size() == 0)
				{
					origin.q = Quat();
				}
				else if (body->joints.size() == 1)
				{
					// Hinge
					body->joints[0]->axis = Normalize(body->joints[0]->axis);
					Vec3 rotVec = Cross(Vec3(1.0f, 0.0f, 0.0f), body->joints[0]->axis);
					if (Length(rotVec) < 1e-5f)
					{
						rotVec = Vec3(0.0f, 1.0f, 0.0f);
					}
					else
					{
						rotVec = Normalize(rotVec);
					}

					float angle = acos(body->joints[0]->axis.x);
					origin.q = QuatFromAxisAngle(rotVec, angle);
				}
				else if (body->joints.size() == 2)
				{
					// Cone
					body->joints[0]->axis = Normalize(body->joints[0]->axis);
					body->joints[1]->axis = Normalize(body->joints[1]->axis);
					if (fabs(Dot(body->joints[0]->axis, body->joints[1]->axis)) > 1e-4)
					{
						cout << "Don't know how to handle non-othogonal joint axis" << endl;
						exit(0);
					}
					Vec3 z = Normalize(Cross(body->joints[1]->axis, body->joints[0]->axis));

					Matrix33 mat = Matrix33(body->joints[1]->axis, body->joints[0]->axis, z);
					origin.q = Quat(mat);
				}
				else if (body->joints.size() == 3)
				{
					// Spherical
					body->joints[0]->axis = Normalize(body->joints[0]->axis);
					body->joints[1]->axis = Normalize(body->joints[1]->axis);
					body->joints[2]->axis = Normalize(body->joints[2]->axis);
					if ((fabs(Dot(body->joints[0]->axis, body->joints[1]->axis)) > 1e-4) ||
						(fabs(Dot(body->joints[0]->axis, body->joints[2]->axis)) > 1e-4) ||
						(fabs(Dot(body->joints[1]->axis, body->joints[2]->axis)) > 1e-4))
					{
						cout << "Don't know how to handle non-othogonal joint axis" << endl;
						exit(0);
					}
					Matrix33 mat = Matrix33(body->joints[2]->axis, body->joints[1]->axis, body->joints[0]->axis);
					origin.q = Quat(mat);
				}
				else
				{
					cout << "Don't know how to handle >3 joints" << endl;
					exit(0);
				}

				Transform bOrigin = origin;
				origin = myTrans*origin;
				Transform ptran = (parent >= 0) ? rigidTrans[parent] : Transform();
				Transform mtran = myTrans;

				Transform ppose = (Inverse(ptran))*origin;
				Transform cpose = (Inverse(mtran))*origin;

				NvFlexRigidJoint joint;
				memset(&joint, 0, sizeof(NvFlexRigidJoint));

				joint.body0 = parent;
				joint.body1 = myIndex;
				joint.maxIterations = INT_MAX;
				joint.flags = eNvFlexRigidJointFlagsDisableCollision;

				joint.pose0 = NvFlexMakeRigidPose(ppose.p, ppose.q);
				joint.pose1 = NvFlexMakeRigidPose(cpose.p, cpose.q);

				// All locked
				for (int k = 0; k < 6; ++k)
				{
					joint.modes[k] = eNvFlexRigidJointModePosition;
				}

				NvFlexRigidJoint springJoint;
				memcpy(&springJoint, &joint, sizeof(NvFlexRigidJoint));
				for (int k = 0; k < 6; ++k)
				{
					springJoint.modes[k] = eNvFlexRigidJointModeFree;
				}

				Matrix33 inertia(g_buffers->rigidBodies.back().inertia);

				// enable limits if set
				NvFlexRigidJointAxis axisHinge[3] = { eNvFlexRigidJointAxisTwist, eNvFlexRigidJointAxisSwing1, eNvFlexRigidJointAxisSwing2 };
				NvFlexRigidJointAxis axisSlide[3] = { eNvFlexRigidJointAxisX, eNvFlexRigidJointAxisY, eNvFlexRigidJointAxisZ };
				int jointIndex = g_buffers->rigidJoints.size();
				int activeJointIndex = jointIndex + 1;
				for (int d = 0; d < (int)body->joints.size(); d++)
				{
					int jid = body->joints.size() - 1 - d;
					if (body->joints[jid]->type == MJCFJoint::HINGE)
					{
						if (body->joints[jid]->limited)
						{
							joint.modes[axisHinge[d]] = eNvFlexRigidJointModeLimit;
							joint.lowerLimits[axisHinge[d]] = body->joints[jid]->range.x;
							joint.upperLimits[axisHinge[d]] = body->joints[jid]->range.y;							
							jmap[body->joints[jid]->name] = make_pair(jointIndex, axisHinge[d]);
							d6jmap[body->joints[jid]->name] = make_pair(g_buffers->rigidJoints.size(), axisHinge[d]);

							// Armature
							Vec3 globalAxis = Rotate(myTrans.q, body->joints[jid]->axis);
							Quat q(g_buffers->rigidBodies.back().theta);
							Vec3 bodyAxis = RotateInv(q, globalAxis);
						//	Matrix33 mat = Outer(bodyAxis, bodyAxis);
							inertia += body->joints[jid]->armature*Outer(bodyAxis, bodyAxis);

							// Joint spring
							if (body->joints[jid]->ref != 0.0f)
							{
								cout << "Don't know how to deal with joint with != 0 ref yet!" << endl;
								exit(0);
							}

                            if (body->joints[jid]->stiffness > 0.0f)
                            {
                                // enable limit spring mode
                                joint.modes[axisHinge[d]] = eNvFlexRigidJointModeLimitSpring;
                                joint.compliance[axisHinge[d]] = 1.0f / std::max(1e-12f, body->joints[jid]->stiffness);
                                joint.damping[axisHinge[d]] = body->joints[jid]->damping;
                            }

							if (createActiveJoints)
							{
								activeJointsNameMap[body->joints[jid]->name] = activeJoints.size();
								activeJoints.push_back(make_pair(activeJointIndex, axisHinge[d]));
							}
						}
						else
						{
							joint.modes[axisHinge[d]] = eNvFlexRigidJointModeFree;
						}
					}
					else if (body->joints[jid]->type == MJCFJoint::SLIDE)
					{
						allHinge = false;
						if (body->joints[jid]->limited)
						{
							if (body->joints[jid]->armature > 0.0f)
							{
								cout << "Does not support armature for sliding joint yet" << endl;
							}
							if ((body->joints[jid]->stiffness > 0.0f) || (body->joints[jid]->damping > 0.0f))
							{
								cout << "Does not support stiffness and damping for sliding joint yet" << endl;
							}
							joint.modes[axisSlide[d]] = eNvFlexRigidJointModeLimit;
							joint.lowerLimits[axisSlide[d]] = body->joints[jid]->range.x;
							joint.upperLimits[axisSlide[d]] = body->joints[jid]->range.y;

							jmap[body->joints[jid]->name] = make_pair(jointIndex, axisSlide[d]);
						}
						else
						{
							joint.modes[axisSlide[d]] = eNvFlexRigidJointModeFree;
						}
					}
				}
				memcpy(&g_buffers->rigidBodies.back().inertia[0], &inertia, sizeof(float) * 9);
				bool succ = false;
				Matrix33 invInertia = Inverse(inertia, succ);
				memcpy(&g_buffers->rigidBodies.back().invInertia[0], &invInertia, sizeof(float) * 9);
				//joint->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, false);

				if (sameOrigin)
				{
					g_buffers->rigidJoints.push_back(joint);
					if (createActiveJoints)
					{
						// Extra active joint
						g_buffers->rigidJoints.push_back(springJoint);
					}
					//g_buffers->rigidJoints.push_back(springJoint);
				}
				else
				{
					cout << "Not same origin" << endl;
					exit(0);
					// Not same origin!
					if ((body->joints.size() != 2) || (!allHinge))
					{
						// Can only 2 joints, all hinge for now
						// Make axis 0 free
						g_buffers->rigidJoints.push_back(joint);
						//g_buffers->rigidJoints.push_back(springJoint);

						joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;
						g_buffers->rigidJoints.push_back(joint);
						springJoint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;
						//g_buffers->rigidJoints.push_back(springJoint);
						origin = bOrigin;
						origin.p = body->joints[1]->pos;
						origin = myTrans*origin;
						ppose = (Inverse(ptran))*origin;
						cpose = (Inverse(mtran))*origin;

						joint.modes[eNvFlexRigidJointAxisSwing1] = eNvFlexRigidJointModeFree;
						springJoint.modes[eNvFlexRigidJointAxisSwing1] = eNvFlexRigidJointModeFree;
						joint.modes[eNvFlexRigidJointAxisSwing2] = eNvFlexRigidJointModeFree;
						springJoint.modes[eNvFlexRigidJointAxisSwing2] = eNvFlexRigidJointModeFree;
						joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeLimit;
						springJoint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;

						joint.pose0 = NvFlexMakeRigidPose(ppose.p, ppose.q);
						joint.pose1 = NvFlexMakeRigidPose(cpose.p, cpose.q);
						springJoint.pose0 = NvFlexMakeRigidPose(ppose.p, ppose.q);
						springJoint.pose1 = NvFlexMakeRigidPose(cpose.p, cpose.q);

						g_buffers->rigidJoints.push_back(joint);
						if (createActiveJoints)
						{
							// Extra active joint
							g_buffers->rigidJoints.push_back(springJoint);
						}
						//g_buffers->rigidJoints.push_back(springJoint);
					}
					else
					{
						cout << "Can't handle!\n" << endl;
					}

				}
				d6jointCounter++;
			}
		}
        // Recursively create children's bodies
        for (int i = 0; i < (int)body->bodies.size(); i++)
        {
            createPhysicsBodyAndJoint(body->bodies[i], myTrans, myIndex, maxAngularVelocity, collideEachOther);
        }
    }

    MJCFImporter(const char* fname)
    {
        d6jointCounter = 0;
        tinyxml2::XMLDocument doc;
        doc.LoadFile(GetFilePathByPlatform(fname).c_str());
		defaultClassName = "main";
        XMLElement* root = doc.RootElement();
        LoadCompiler(root->FirstChildElement("compiler"));
        
		// Deal with defaults
		XMLElement* d =root->FirstChildElement("default");
		if (!d)
		{
			// No default, set the defaultClassName to default....
			classes[defaultClassName] = MJCFClass();
		}
		else
		{
			// Only handle one top level default 
			if (d->Attribute("class")) defaultClassName = d->Attribute("class");
			classes[defaultClassName] = MJCFClass();
			LoadDefault(d, defaultClassName, classes[defaultClassName]);
			if (d->NextSiblingElement("default"))
			{
				cout << "Can only handle one top level default at the moment!" << endl;
				exit(0);
			}
		}

		XMLElement* a = root->FirstChildElement("asset");
		if (a)
		{
			XMLElement* m = a->FirstChildElement("mesh");
			while (m)
			{
				std::string meshName;
				std::string meshFile;

				getIfExist(m, "name", meshName);
				getIfExist(m, "file", meshFile);

				char meshPath[2048];
				MakeRelativePath(fname, (compiler.meshDir + meshFile).c_str(), meshPath);

				Mesh* mesh = ImportMesh(meshPath);
				if (mesh)
				{
					const float dilation = 0.005f;

					NvFlexTriangleMeshId meshId = CreateTriangleMesh(mesh, dilation);
					assets[meshName] = meshId;

					cout << "Imported mesh file: " << meshFile << " from mesh dir: " << compiler.meshDir << endl;
				}
				else
				{
					cout << "Failed to read mesh file: " << meshFile << " from mesh dir: " << compiler.meshDir << endl;
				}

				m = m->NextSiblingElement("mesh");
			}
		}
        XMLElement* wb = root->FirstChildElement("worldbody");
        XMLElement* c = wb->FirstChildElement("body");
        while (c)
        {
            bodies.push_back(new MJCFBody());
            LoadBody(c, *bodies.back(), defaultClassName);
            c = c->NextSiblingElement("body");
        }

        XMLElement* ac = root->FirstChildElement("actuator");
        if (ac)
        {
            c = ac->FirstChildElement("motor");
            while (c)
            {
                motors.push_back(new MJCFMotor());
                LoadMotor(c, *motors.back(), defaultClassName);
                c = c->NextSiblingElement("motor");
            }
        }

		for (int i = 0; i < (int)motors.size(); i++)
		{
			controlMap[motors[i]->joint] = i;
		}
    }

	void AddPhysicsEntities(Transform trans, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& motorPower, float maxAngularVelocity = 64.f,
							bool collideEachOther = true, bool createActiveJoints = false, bool createBodyForFixedJoint = true)
	{
		activeJointsNameMap.clear();
		d6jmap.clear();
		activeJoints.clear(); // Joint index, dof index
		activeJointsNameMap.clear(); // Map name to index in activeJoints
		bmap.clear();
		jmap.clear();
		d6jmap.clear();

		rigidTrans.clear();
		myShapes.clear();
		geoBodyPose.clear();

		this->createActiveJoints = createActiveJoints;
		this->createBodyForFixedJoint = createBodyForFixedJoint;

		firstBody = g_buffers->rigidBodies.size();
		for (int i = 0; i < (int)bodies.size(); i++)
		{
			createPhysicsBodyAndJoint(bodies[i], trans, -1, maxAngularVelocity, collideEachOther);
		}

		for (int i = 0; i < (int)motors.size(); i++)
		{
			ctrl.push_back(d6jmap[motors[i]->joint]);
			motorPower.push_back(motors[i]->gear);
		}
	}

	void setBodiesByAngles(MJCFBody* body, Transform trans, float* angles)
	{
		Transform myTrans = trans * Transform(body->pos, body->quat);
		Quat q;
		// Now apply random rotation
		for (int d = 0; d < (int)body->joints.size(); d++)
		{
			int jid = body->joints.size() - 1 - d;
			if (body->joints[jid]->type == MJCFJoint::HINGE)
			{
				
				Quat myQ = QuatFromAxisAngle(body->joints[jid]->axis, angles[controlMap[body->joints[d]->name]]);
				q = q * myQ;
			}
		}
		Transform rot(Vec3(0.0f, 0.0f, 0.0f), q);
		int mi = bmap[body->name];
		myTrans = myTrans*rot;
		NvFlexSetRigidPose(&g_buffers->rigidBodies[mi], (NvFlexRigidPose*)&myTrans);

		Vec3 v(0.0f, 0.0f, 0.0f);
		memcpy(&g_buffers->rigidBodies[mi].linearVel, &v, sizeof(float) * 3);
		Vec3 av(0.0f, 0.0f, 0.0f);
		memcpy(&g_buffers->rigidBodies[mi].angularVel, &av, sizeof(float) * 3);

		memset(&g_buffers->rigidBodies[mi].force, 0, sizeof(float) * 3);
		memset(&g_buffers->rigidBodies[mi].torque, 0, sizeof(float) * 3);
		for (int i = 0; i < (int)body->bodies.size(); i++)
		{
			setBodiesByAngles(body->bodies[i], myTrans, angles);
		}
	}

	void setBodiesByAngles(Transform trans, float* angles)
	{
		for (int i = 0; i < (int)bodies.size(); i++)
		{
			setBodiesByAngles(bodies[i], trans, angles);
		}
	}

    void resetBodies(MJCFBody* body, Transform trans, float angleNoise, float velNoise, float angularVelNoise)
    {
        Transform myTrans = trans * Transform(body->pos, body->quat);
        Quat q;
        // Now apply random rotation
        for (int d = 0; d < (int)body->joints.size(); d++)
        {
            int jid = body->joints.size() - 1 - d;
            if (body->joints[jid]->type == MJCFJoint::HINGE)
            {
                Quat myQ = QuatFromAxisAngle(body->joints[jid]->axis, (Randf() - 0.5f) * 2.f * angleNoise);
                q = q * myQ;
            }
        }

        Transform rot(Vec3(0.0f, 0.0f, 0.0f), q);
        int mi = bmap[body->name];
        myTrans = myTrans * rot;
        NvFlexSetRigidPose(&g_buffers->rigidBodies[mi], (NvFlexRigidPose*)&myTrans);

        Vec3 v = RandomUnitVector() * velNoise;
        memcpy(&g_buffers->rigidBodies[mi].linearVel, &v, sizeof(float) * 3);
        Vec3 av = RandomUnitVector() * angularVelNoise;
        memcpy(&g_buffers->rigidBodies[mi].angularVel, &av, sizeof(float) * 3);

        memset(&g_buffers->rigidBodies[mi].force, 0, sizeof(float) * 3);
        memset(&g_buffers->rigidBodies[mi].torque, 0, sizeof(float) * 3);
        for (int i = 0; i < (int)body->bodies.size(); i++)
        {
            resetBodies(body->bodies[i], myTrans, angleNoise, velNoise, angularVelNoise);
        }
    }

    void reset(Transform trans, float angleNoise, float velNoise, float angularVelNoise)
    {
        for (int i = 0; i < (int)bodies.size(); i++)
        {
            resetBodies(bodies[i], trans, angleNoise, velNoise, angularVelNoise);
        }
    }

	void storeJointAngleNoise(MJCFBody* body, float angleNoise, float velNoise, float angularVelNoise)
	{
		int prevIdx = -1;
		Vec3 prevPos;
		float prevTwist;
		float prevSwing1;
		float prevSwing2;
		for (int d = 0; d < (int)body->joints.size(); d++)
		{
			pair<int, NvFlexRigidJointAxis> jj = jmap[body->joints[d]->name];
			if (body->joints[d]->type == MJCFJoint::HINGE)
			{
				int gj = jj.first;

				NvFlexRigidJoint& joint = g_buffers->rigidJoints[gj];
				if (gj != prevIdx)
				{
					NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
					NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

					Transform body0Pose;
					NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
					Transform body1Pose;
					NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

					Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
					Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
					Transform relPose = Inverse(pose0)*pose1;

					prevPos = relPose.p;
					Quat qd = relPose.q;

					Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
					Quat qswing = qd*Inverse(qtwist);
					prevTwist = asin(qtwist.x)*2.0f;
					prevSwing1 = asin(qswing.y)*2.0f;
					prevSwing2 = asin(qswing.z)*2.0f;
					prevIdx = gj;

					// If same, no need to recompute
				}
				if (jj.second == eNvFlexRigidJointAxisTwist) body->joints[d]->initVal = prevTwist;
				if (jj.second == eNvFlexRigidJointAxisSwing1) body->joints[d]->initVal = prevSwing1;
				if (jj.second == eNvFlexRigidJointAxisSwing2) body->joints[d]->initVal = prevSwing2;
				if ((jj.second == eNvFlexRigidJointAxisX) ||
					(jj.second == eNvFlexRigidJointAxisY) ||
					(jj.second == eNvFlexRigidJointAxisZ)) {
					//cout << "Not support sliding joint yet " << d<<" "<<jj.second<<endl;
				}
			}
		}

		for (int i = 0; i < (int)body->bodies.size(); i++)
		{
			storeJointAngleNoise(body->bodies[i], angleNoise, velNoise, angularVelNoise);
		}
	}

	void applyBodies(MJCFBody* body, Transform trans, float angleNoise, float velNoise, float angularVelNoise)
	{
		Transform myTrans = trans*Transform(body->pos, body->quat);
		Quat q;
		// Now apply random rotation
		for (int d = 0; d < (int)body->joints.size(); d++)
		{
			int jid = body->joints.size() - 1 - d;
			if (body->joints[jid]->type == MJCFJoint::HINGE)
			{
				Quat myQ = QuatFromAxisAngle(body->joints[jid]->axis, (Randf() - 0.5f) * 2.f * angleNoise + body->joints[jid]->initVal);
				q = q * myQ;
			}
		}
		Transform rot(Vec3(0.0f, 0.0f, 0.0f), q);
		int mi = bmap[body->name];
		myTrans = myTrans*rot;
		NvFlexSetRigidPose(&g_buffers->rigidBodies[mi], (NvFlexRigidPose*)&myTrans);

		Vec3 v = RandomUnitVector() * velNoise + (Vec3&)g_buffers->rigidBodies[mi].linearVel;
		memcpy(&g_buffers->rigidBodies[mi].linearVel, &v, sizeof(float) * 3);
		Vec3 av = RandomUnitVector() * angularVelNoise + (Vec3&)g_buffers->rigidBodies[mi].angularVel;
		memcpy(&g_buffers->rigidBodies[mi].angularVel, &av, sizeof(float) * 3);

		memset(&g_buffers->rigidBodies[mi].force, 0, sizeof(float) * 3);
		memset(&g_buffers->rigidBodies[mi].torque, 0, sizeof(float) * 3);
		for (int i = 0; i < (int)body->bodies.size(); i++)
		{
			applyBodies(body->bodies[i], myTrans, angleNoise, velNoise, angularVelNoise);
		}
	}

	void applyJointAngleNoise(float angleNoise, float velNoise, float angularVelNoise)
	{
		for (int i = 0; i < (int)bodies.size(); i++)
		{
			storeJointAngleNoise(bodies[i], angleNoise, velNoise, angularVelNoise);
		}
		Transform myTrans;
		
		NvFlexGetRigidPose(&g_buffers->rigidBodies[bmap[bodies[0]->name]], (NvFlexRigidPose*)&myTrans);
		Transform trans = myTrans * Inverse(Transform(bodies[0]->pos, bodies[0]->quat)); // Extract root transform 

		// Now restore angles
		for (int i = 0; i < (int)bodies.size(); i++)
		{
			applyBodies(bodies[i], trans, angleNoise, velNoise, angularVelNoise);
		}
	}
};
