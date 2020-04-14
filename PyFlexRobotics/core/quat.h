// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2016 NVIDIA Corporation. All rights reserved.

#pragma once

#include <cassert>

struct Matrix33;

template <typename T>
class XQuat
{
public:

	typedef T value_type;

	CUDA_CALLABLE XQuat() : x(0), y(0), z(0), w(1.0) {}
	CUDA_CALLABLE XQuat(const T* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}
	CUDA_CALLABLE XQuat(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) { 	}
	CUDA_CALLABLE XQuat(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) { }
	CUDA_CALLABLE explicit XQuat(const Matrix33& m);

	CUDA_CALLABLE operator T* () { return &x; }
	CUDA_CALLABLE operator const T* () const { return &x; };

	CUDA_CALLABLE void Set(T x_, T y_, T z_, T w_) {  x = x_; y = y_; z = z_; w = w_; }

	CUDA_CALLABLE XQuat<T> operator * (T scale) const { XQuat<T> r(*this); r *= scale;  return r;}
	CUDA_CALLABLE XQuat<T> operator / (T scale) const { XQuat<T> r(*this); r /= scale;  return r; }
	CUDA_CALLABLE XQuat<T> operator + (const XQuat<T>& v) const { XQuat<T> r(*this); r += v;  return r; }
	CUDA_CALLABLE XQuat<T> operator - (const XQuat<T>& v) const { XQuat<T> r(*this); r -= v;  return r; }
	CUDA_CALLABLE XQuat<T> operator * (XQuat<T> q) const 
	{
		// quaternion multiplication
		return XQuat<T>(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
		            w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);		
	}

	CUDA_CALLABLE XQuat<T>& operator *=(T scale) {x *= scale; y *= scale; z*= scale; w*= scale;  return *this;}
	CUDA_CALLABLE XQuat<T>& operator /=(T scale) {T s(1.0f/scale); x *= s; y *= s; z *= s; w *=s;  return *this;}
	CUDA_CALLABLE XQuat<T>& operator +=(const XQuat<T>& v) {x += v.x; y += v.y; z += v.z; w += v.w;  return *this;}
	CUDA_CALLABLE XQuat<T>& operator -=(const XQuat<T>& v) {x -= v.x; y -= v.y; z -= v.z; w -= v.w;  return *this;}

	CUDA_CALLABLE bool operator != (const XQuat<T>& v) const { return (x != v.x || y != v.y || z != v.z || w != v.w); }

	// negate
	CUDA_CALLABLE XQuat<T> operator -() const {  return XQuat<T>(-x, -y, -z, -w); }

	CUDA_CALLABLE XVector3<T> GetAxis() const { return XVector3<T>(x, y, z); }

	T x,y,z,w;
};

typedef XQuat<float> Quat;

// lhs scalar scale
template <typename T>
CUDA_CALLABLE XQuat<T> operator *(T lhs, const XQuat<T>& rhs)
{
	XQuat<T> r(rhs);
	r *= lhs;
	return r;
}

template <typename T>
CUDA_CALLABLE bool operator==(const XQuat<T>& lhs, const XQuat<T>& rhs)
{
	return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

template <typename T>
CUDA_CALLABLE inline XQuat<T> QuatFromAxisAngle(const Vec3& axis, float angle)
{
	Vec3 v = Normalize(axis);

	float half = angle*0.5f;
	float w = cosf(half);

	const float sin_theta_over_two = sinf(half);
	v *= sin_theta_over_two;

	return XQuat<T>(v.x, v.y, v.z, w);
}

CUDA_CALLABLE inline float Dot(const Quat& a, const Quat& b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

CUDA_CALLABLE inline float Length(const Quat& a)
{
	return sqrtf(Dot(a, a));
}

CUDA_CALLABLE inline Quat rpy2quat(float roll, float pitch, float yaw)
{
	Quat q;
	// Abbreviations for the various angular functions
	float cy = cos(yaw * 0.5f);
	float sy = sin(yaw * 0.5f);
	float cr = cos(roll * 0.5f);
	float sr = sin(roll * 0.5f);
	float cp = cos(pitch * 0.5f);
	float sp = sin(pitch * 0.5f);

	q.w = (float)(cy * cr * cp + sy * sr * sp);
	q.x = (float)(cy * sr * cp - sy * cr * sp);
	q.y = (float)(cy * cr * sp + sy * sr * cp);
	q.z = (float)(sy * cr * cp - cy * sr * sp);

	return q;
}

CUDA_CALLABLE inline void quat2rpy(const Quat& q1, float& bank, float& attitude, float& heading)
{
	float sqw = q1.w * q1.w;
	float sqx = q1.x * q1.x;
	float sqy = q1.y * q1.y;
	float sqz = q1.z * q1.z;
	float unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
	float test = q1.x*q1.y + q1.z*q1.w;

	if (test > 0.499f * unit)
	{ // singularity at north pole
		heading = 2.f * atan2(q1.x, q1.w);
		attitude = kPi / 2.f;
		bank = 0.f;
		return;
	}

	if (test < -0.499f * unit)
	{ // singularity at south pole
		heading = -2.f * atan2(q1.x, q1.w);
		attitude = -kPi / 2.f;
		bank = 0.f;
		return;
	}

	heading = atan2(2.f * q1.y*q1.w - 2.f * q1.x*q1.z, sqx - sqy - sqz + sqw);
	attitude = asin(2.f * test / unit);
	bank = atan2(2.f * q1.x*q1.w - 2.f * q1.y*q1.z, -sqx + sqy - sqz + sqw);
}

// rotate vector by quaternion (q, w)
CUDA_CALLABLE inline Vec3 Rotate(const Quat& q, const Vec3& x)
{
	return x*(2.0f*q.w*q.w-1.0f) + Cross(Vec3(q), x)*q.w*2.0f + Vec3(q)*Dot(Vec3(q), x)*2.0f;
}

CUDA_CALLABLE inline Vec3 operator*(const Quat& q, const Vec3& v)
{
	return Rotate(q, v);
}

CUDA_CALLABLE inline Vec3 GetBasisVector0(const Quat& q)
{
	return Rotate(q, Vec3(1.0f, 0.0f, 0.0f));
}
CUDA_CALLABLE inline Vec3 GetBasisVector1(const Quat& q)
{
	return Rotate(q, Vec3(0.0f, 1.0f, 0.0f));
}
CUDA_CALLABLE inline Vec3 GetBasisVector2(const Quat& q)
{
	return Rotate(q, Vec3(0.0f, 0.0f, 1.0f));
}

// rotate vector by inverse transform in (q, w)
CUDA_CALLABLE inline Vec3 RotateInv(const Quat& q, const Vec3& x)
{
	return x*(2.0f*q.w*q.w-1.0f) - Cross(Vec3(q), x)*q.w*2.0f + Vec3(q)*Dot(Vec3(q), x)*2.0f;
}

CUDA_CALLABLE inline Quat Inverse(const Quat& q)
{
	return Quat(-q.x, -q.y, -q.z, q.w);
}

CUDA_CALLABLE inline Quat Normalize(const Quat& q)
{
	float lSq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;

	if (lSq > 0.0f)
	{
		float invL = 1.0f / sqrtf(lSq);

		return q*invL;
	}
	else
		return Quat();
}

//
// given two quaternions and a time-step returns the corresponding angular velocity vector
//
CUDA_CALLABLE inline Vec3 DifferentiateQuat(const Quat& q1, const Quat& q0, float invdt)
{
	Quat dq = q1*Inverse(q0);

	float sinHalfTheta = Length(dq.GetAxis());
	float theta = asinf(sinHalfTheta)*2.0f;
	
	if (fabsf(theta) < 0.001f)
	{
		// use linear approximation approx for small angles
		Quat dqdt = (q1-q0)*invdt;
		Quat omega = dqdt*Inverse(q0);

		return Vec3(omega.x, omega.y, omega.z)*2.0f;	
	}
	else
	{
		// use inverse exponential map
		Vec3 axis = Normalize(dq.GetAxis());
		return axis*theta*invdt;	
	}
}


CUDA_CALLABLE inline Quat IntegrateQuat(const Vec3& omega, const Quat& q0, float dt)
{	
	Vec3 axis;
	float w = Length(omega);

	if (w*dt < 0.001f)
	{
		// sinc approx for small angles
		axis = omega*(0.5f*dt-(dt*dt*dt)/48.0f*w*w);
	}
	else
	{
		axis = omega*(sinf(0.5f*w*dt)/w);
	}

	Quat dq;   
	dq.x = axis.x;
	dq.y = axis.y;
	dq.z = axis.z;
	dq.w = cosf(w*dt*0.5f);
	 
	Quat q1 = dq*q0;

	// explicit re-normalization here otherwise we do some see energy drift
	return Normalize(q1);
}
