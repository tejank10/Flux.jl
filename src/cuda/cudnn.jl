using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, cudnnTensorDescriptor_t,
  cudnnBatchNormMode_t, cudnnHandle_t, libcudnn_handle, cudnnDataType, TensorDesc, FilterDesc
using CuArrays
using Flux
mutable struct DropoutDesc
  ptr::Ptr{Void}
  states::CuVector{UInt8}
end

Base.unsafe_convert(::Type{Ptr{Void}}, dd::DropoutDesc) = dd.ptr

function DropoutDesc(ρ::Real; seed::Integer=0)
  d = [C_NULL]
  s = Csize_t[0]
  @check ccall((:cudnnCreateDropoutDescriptor,libcudnn), cudnnStatus_t, (Ptr{Ptr{Void}},), d)
  @check ccall((:cudnnDropoutGetStatesSize,libcudnn),cudnnStatus_t,(Ptr{Void},Ptr{Csize_t}),libcudnn_handle[],s)
  states = CuArray{UInt8}(s[]) # TODO: can we drop this when ρ=0?
  desc = DropoutDesc(d[], states)
  @check ccall((:cudnnSetDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Void},Ptr{Void},Cfloat,Ptr{Void},Csize_t,Culonglong),
    desc,libcudnn_handle[],ρ,states,length(states),seed)
  finalizer(desc, x ->
    @check ccall((:cudnnDestroyDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Void},),x))
  return desc
end

const BATCHNORM_SPATIAL = 1
const BATCHNORM_ACTIVATION = 0
const BATCHNORM_MIN_EPS = 1e-5

bnshape(x::NTuple{4}) = x
bnshape(x::Union{NTuple{1},NTuple{2},NTuple{3}}) = bnshape((1,x...))
bnshape(x::AbstractArray) = bnshape(size(x))

function batchnorm(x::CuArray{T}) where T<:Union{Float32,Float64}
  y = similar(x)
  sh = bnshape(x)
  td_x = TensorDesc(T, sh)
  td_p = TensorDesc(T, (1,1,sh[3],1))
  # @check ccall((:cudnnBatchNormalizationForwardTraining,libcudnn),cudnnStatus_t,
    # (Ptr{Void}, UInt32,
     # Ptr{T}, Ptr{T}, #alpha and beta
     # Ptr{Void}, Ptr{T}, #xdesc and x
     # Ptr{Void}, Ptr{T}, #ydesc and y
     # Ptr{Void}, Ptr{T}, Ptr{T}, #desc, weight and bias
    #  Cdouble, Ptr{T}, Ptr{T}, #Decay factor, Running mean and Running var
    # Cdouble, # eps
    # Ptr{T}, Ptr{T}), #Cached mean and ivar
     cudnnBatchNormalizationForwardTraining(libcudnn_handle[], BATCHNORM_SPATIAL,
     Ref(T(1)), Ref(T(0)),
     TensorDesc(x), x, #x
     TensorDesc(y), y, #y
     TensorDesc(g), g, b, #params
     momentum, running_mean, running_var,
     eps, mean, ivar)
     return y
end

#=
cudnnStatus_t cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t         derivedBnDesc,
    const cudnnTensorDescriptor_t   xDesc,
    cudnnBatchNormMode_t            mode)
    =#
function cudnnDeriveBNTensorDescriptor(derivedBnDesc::TensorDesc, xDesc, mode)
  @check ccall((:cudnnDeriveBNTensorDescriptor, libcudnn), cudnnStatus_t,
    (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t), derivedBnDesc, xDesc, mode)
end

#=
cudnnStatus_t cudnnBatchNormalizationBackward(
      cudnnHandle_t                    handle,
      cudnnBatchNormMode_t             mode,
      const void                      *alphaDataDiff,
      const void                      *betaDataDiff,
      const void                      *alphaParamDiff,
      const void                      *betaParamDiff,
      const cudnnTensorDescriptor_t    xDesc,
      const void                      *x,
      const cudnnTensorDescriptor_t    dyDesc,
      const void                      *dy,
      const cudnnTensorDescriptor_t    dxDesc,
      void                            *dx,
      const cudnnTensorDescriptor_t    bnScaleBiasDiffDesc,
      const void                      *bnScale,
      void                            *resultBnScaleDiff,
      void                            *resultBnBiasDiff,
      double                           epsilon,
      const void                      *savedMean,
      const void                      *savedInvVariance)
      =#
function cudnnBatchNormalizationBackward(handle,mode,alphaDataDiff,betaDataDiff,
  alphaParamDiff,betaParamDiff,xDesc,x,dyDesc,dy,dxDesc,dx,bnScaleBiasDiffDesc,
  bnScale,resultBnScaleDiff,resultBnBiasDiff,epsilon,savedMean,savedInvVariance)

  @check ccall((:cudnnBatchNormalizationBackward, libcudnn),cudnnStatus_t,
    (cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
     cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},
     cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},
     Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void}),
     handle,mode,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,
     xDesc,x,dyDesc,dy,dxDesc,dx,bnScaleBiasDiffDesc,bnScale,
     resultBnScaleDiff,resultBnBiasDiff,epsilon,savedMean,savedInvVariance)
end

#=
cudnnStatus_t cudnnBatchNormalizationForwardInference(
      cudnnHandle_t                    handle,
      cudnnBatchNormMode_t             mode,
      const void                      *alpha,
      const void                      *beta,
      const cudnnTensorDescriptor_t    xDesc,
      const void                      *x,
      const cudnnTensorDescriptor_t    yDesc,
      void                            *y,
      const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
      const void                      *bnScale,
      const void                      *bnBias,
      const void                      *estimatedMean,
      const void                      *estimatedVariance,
      double                           epsilon)
      =#
function cudnnBatchNormalizationForwardInference(handle,mode,alpha,beta,xDesc,x,yDesc,y,
  bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,
  estimatedMean,estimatedVariance,epsilon)

  @check ccall((:cudnnBatchNormalizationForwardInference, libcudnn), cudnnStatus_t,
  (cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void}, Ptr{Void},
   cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},
   cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void},
   Cdouble,Ptr{Void},Ptr{Void}),handle,mode,alpha,beta,xDesc,x,yDesc,y,
     bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedVariance,epsilon)
end

#=
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
      cudnnHandle_t                    handle,
      cudnnBatchNormMode_t             mode,
      const void                      *alpha,
      const void                      *beta,
      const cudnnTensorDescriptor_t    xDesc,
      const void                      *x,
      const cudnnTensorDescriptor_t    yDesc,
      void                            *y,
      const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
      const void                      *bnScale,
      const void                      *bnBias,
      double                           exponentialAverageFactor,
      void                            *resultRunningMean,
      void                            *resultRunningVariance,
      double                           epsilon,
      void                            *resultSaveMean,
      void                            *resultSaveInvVariance)
      =#

function cudnnBatchNormalizationForwardTraining!(handle,mode,alpha,beta,xDesc,x,
  yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,
  resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,
  resultSaveInvVariance)

  T = eltype(x)

  @check ccall((:cudnnBatchNormalizationForwardTraining, libcudnn), cudnnStatus_t,
  (cudnnHandle_t,UInt32,Ptr{Void}, Ptr{Void},
   Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},
   Ptr{Void},Ptr{Void},Ptr{Void},
   Cdouble,Ptr{Void},Ptr{Void},
   Cdouble,Ptr{Void},Ptr{Void}),
   handle,UInt32(mode),Ref(T(alpha)),Ref(T(beta)),xDesc,x,yDesc,y,
   bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,
   resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,
   resultSaveInvVariance)
end

CuParam{T,N} = Union{CuArray{T,N},TrackedArray{T,N,CuArray{T,N}}}
CuBatchNorm{T} = Flux.BatchNorm{<:Union{typeof(identity),typeof(relu)},
                                <:CuParam{T,1},<:CuArray{T,1},
                                <:Union{Float32,Float64}}

CuBatchNorm(chs::Integer, λ = identity;
			          initβ = zeros, initγ = ones, ϵ = 1e-8, momentum = .1) =
  BatchNorm(λ, param(cu(initβ(Float32,chs))), param(cu(initγ(Float32,chs))),
	                   zeros(Float32,chs), ones(Float32,chs), ϵ, momentum, true)
function (CuBN::CuBatchNorm{T})(x::CuArray{T,4};handle=libcudnn_handle[],
                                alpha=1, beta=0,exponentialAverageFactor=T(1),
                                mode=CUDNN_BATCHNORM_SPATIAL,
                                ϵ=CUDNN_BN_MIN_EPSILON) where T

  y = similar(x)
  xDesc = TensorDesc(x)
  yDesc = TensorDesc(y)
  bnScaleBiasMeanVarDesc = TensorDesc(CuBN.γ)
  resultSaveMean = similar(CuBN.γ)
  resultSaveInvVariance = similar(CuBN.γ)
  handle = handle(x)

  cudnnBatchNormalizationForwardTraining!(
  handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,CuBN.γ, CuBN.β,
  exponentialAverageFactor,CuBN.μ,CuBN.σ,ϵ,
  resultSaveMean, resultSaveInvVariance)

  return y
end
#=
function batchnorm_train(x::CuArray{T,4}, s::BatchNormState; opts...) where T
  y = similar(x)
  batchnorm_train!(y, x, s;  opts...)
  return y
end
=#
a = CuBatchNorm(4)
b = BatchNorm(4)
inp = randn(10,10,4,1)
cuinp = cu(inp)
@time a(cuinp) # 7.127122 seconds (2.52 M allocations: 130.674 MiB, 0.89% gc time)
@time b(inp) # 1.908837 seconds (778.18 k allocations: 36.005 MiB, 0.48% gc time)

#batchnorm(cu(randn(10,5)))

TensorDesc(Float32, (1,5,1,1))

methods(TensorDesc)
