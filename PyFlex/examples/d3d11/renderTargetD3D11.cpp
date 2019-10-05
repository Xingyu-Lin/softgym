/*
 * Copyright (c) 2008-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "renderTargetD3D11.h"

RenderTargetD3D11::RenderTargetD3D11()
{
}

static D3D11_TEXTURE2D_DESC _getTextureDesc(DXGI_FORMAT format, UINT width, UINT height,
	UINT bindFlags, UINT sampleCount = 1, D3D11_USAGE usage = D3D11_USAGE_DEFAULT, UINT cpuAccessFlags = 0,
	UINT miscFlags = 0, UINT arraySize = 1, UINT mipLevels = 1)
{
	D3D11_TEXTURE2D_DESC desc;
	desc.Format = format;
	desc.Width = width;
	desc.Height = height;

	desc.ArraySize = arraySize;
	desc.MiscFlags = miscFlags;
	desc.MipLevels = mipLevels;

	desc.SampleDesc.Count = sampleCount;
	desc.SampleDesc.Quality = 0;
	desc.BindFlags = bindFlags;
	desc.Usage = usage;
	desc.CPUAccessFlags = cpuAccessFlags;
	return desc;
}

static D3D11_DEPTH_STENCIL_VIEW_DESC _getDsvDesc(DXGI_FORMAT format, D3D11_DSV_DIMENSION viewDimension, UINT flags = 0, UINT mipSlice = 0)
{
	D3D11_DEPTH_STENCIL_VIEW_DESC desc;
	desc.Format = format;
	desc.ViewDimension = viewDimension;
	desc.Flags = flags;
	desc.Texture2D.MipSlice = mipSlice;
	return desc;
}

static D3D11_SHADER_RESOURCE_VIEW_DESC _getSrvDesc(DXGI_FORMAT format)
{
	D3D11_SHADER_RESOURCE_VIEW_DESC desc;
	desc.Format = format;
	desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	desc.Texture2D.MostDetailedMip = 0;
	desc.Texture2D.MipLevels = -1;

	return desc;
}

HRESULT RenderTargetD3D11::init(ID3D11Device* device, int width, int height, bool depthTest)
{
	// set viewport
	{
		m_viewport.Width = float(width);
		m_viewport.Height = float(height);
		m_viewport.MinDepth = 0;
		m_viewport.MaxDepth = 1;
		m_viewport.TopLeftX = 0;
		m_viewport.TopLeftY = 0;
	}

	// create shadow render target
	{
		D3D11_TEXTURE2D_DESC texDesc = _getTextureDesc(DXGI_FORMAT_R32_FLOAT, UINT(width), UINT(height),
			D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);

		NV_RETURN_ON_FAIL(device->CreateTexture2D(&texDesc, NV_NULL, m_backTexture.ReleaseAndGetAddressOf()));
		NV_RETURN_ON_FAIL(device->CreateShaderResourceView(m_backTexture.Get(), NV_NULL, m_backSrv.ReleaseAndGetAddressOf()));
		NV_RETURN_ON_FAIL(device->CreateRenderTargetView(m_backTexture.Get(), NV_NULL, m_backRtv.ReleaseAndGetAddressOf()));
	}

	// create shadow depth stencil
	{
		D3D11_TEXTURE2D_DESC texDesc = _getTextureDesc(DXGI_FORMAT_R32_TYPELESS, UINT(width), UINT(height),
			D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE);
		NV_RETURN_ON_FAIL(device->CreateTexture2D(&texDesc, NV_NULL, m_depthTexture.ReleaseAndGetAddressOf()));

		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = _getDsvDesc(DXGI_FORMAT_D32_FLOAT, D3D11_DSV_DIMENSION_TEXTURE2D);
		NV_RETURN_ON_FAIL(device->CreateDepthStencilView(m_depthTexture.Get(), &dsvDesc, m_depthDsv.ReleaseAndGetAddressOf()));

		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = _getSrvDesc(DXGI_FORMAT_R32_FLOAT);
		NV_RETURN_ON_FAIL(device->CreateShaderResourceView(m_depthTexture.Get(), &srvDesc, m_depthSrv.ReleaseAndGetAddressOf()));
	}

	{
		D3D11_SAMPLER_DESC samplerDesc =
		{
			D3D11_FILTER_MIN_MAG_MIP_LINEAR,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			0.0, 0, D3D11_COMPARISON_NEVER,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, D3D11_FLOAT32_MAX,
		};
		NV_RETURN_ON_FAIL(device->CreateSamplerState(&samplerDesc, m_linearSampler.ReleaseAndGetAddressOf()));
	}
	{
		D3D11_SAMPLER_DESC samplerDesc =
		{
			D3D11_FILTER_MIN_MAG_MIP_POINT,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			D3D11_TEXTURE_ADDRESS_CLAMP,
			0.0, 0, D3D11_COMPARISON_NEVER,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, D3D11_FLOAT32_MAX,
		};
		NV_RETURN_ON_FAIL(device->CreateSamplerState(&samplerDesc, m_pointSampler.ReleaseAndGetAddressOf()));
	}

	{
		D3D11_DEPTH_STENCIL_DESC depthStateDesc = {};
		depthStateDesc.DepthEnable = depthTest;
		depthStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		depthStateDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

		NV_RETURN_ON_FAIL(device->CreateDepthStencilState(&depthStateDesc, m_depthStencilState.ReleaseAndGetAddressOf()));
	}

	return S_OK;
}

void RenderTargetD3D11::bindAndClear(ID3D11DeviceContext* context)
{
	ID3D11RenderTargetView* ppRtv[1] = { m_backRtv.Get() };
	context->OMSetRenderTargets(1, ppRtv, m_depthDsv.Get());
	context->OMSetDepthStencilState(m_depthStencilState.Get(), 0u);

	context->RSSetViewports(1, &m_viewport);

	float clearDepth = 0.0f;
	float clearColor[4] = { clearDepth, clearDepth, clearDepth, clearDepth };

	context->ClearRenderTargetView(m_backRtv.Get(), clearColor);
	context->ClearDepthStencilView(m_depthDsv.Get(), D3D11_CLEAR_DEPTH, 1.0, 0);
}
