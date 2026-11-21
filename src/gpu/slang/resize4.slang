// Resize bilinéaire demi-pixel ×4. Les rangées 2k et 2k+1 ont les mêmes voisins.
struct ResizePush
{
    uint inputN, inputC, inputH, inputW, inputOrder, inputPitch;
    uint outputN, outputC, outputH, outputW, outputOrder, outputPitch;
};

[[vk::push_constant]] ConstantBuffer<ResizePush> params;
[[vk::binding(0, 0)]] StructuredBuffer<half> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> outputData;

uint address(uint batch, uint channel, uint y, uint x, uint h, uint w, uint order, uint pitch)
{
    if (order == 0)
        return ((batch * params.inputC + channel) * h + y) * w + x;
    return ((batch * h + y) * w + x) * pitch + channel;
}

half storeHalf(float value)
{
    return half(isnan(value) ? value : clamp(value, -65504.0f, 65504.0f));
}

[numthreads(128, 1, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.outputW * params.outputC)
        return;
    uint channel = params.outputOrder == 0 ? id.x / params.outputW : id.x % params.outputC;
    uint x = params.outputOrder == 0 ? id.x % params.outputW : id.x / params.outputC;
    uint y = id.y * 2;
    precise float sx = clamp((float(x) + 0.5f) * 0.25f - 0.5f, 0.0f, float(params.inputW - 1));
    precise float sy0 = clamp((float(y) + 0.5f) * 0.25f - 0.5f, 0.0f, float(params.inputH - 1));
    precise float sy1 = clamp((float(y + 1) + 0.5f) * 0.25f - 0.5f, 0.0f, float(params.inputH - 1));
    uint left = uint(sx);
    uint top = uint(sy0);
    uint right = min(left + 1, params.inputW - 1);
    uint bottom = min(top + 1, params.inputH - 1);
    precise float wx = sx - float(left);
    precise float wy0 = sy0 - float(top);
    precise float wy1 = sy1 - float(top);
    float a = inputData[address(id.z, channel, top, left, params.inputH, params.inputW, params.inputOrder, params.inputPitch)];
    float b = inputData[address(id.z, channel, top, right, params.inputH, params.inputW, params.inputOrder, params.inputPitch)];
    float c = inputData[address(id.z, channel, bottom, left, params.inputH, params.inputW, params.inputOrder, params.inputPitch)];
    float d = inputData[address(id.z, channel, bottom, right, params.inputH, params.inputW, params.inputOrder, params.inputPitch)];
    precise float first = (1.0f - wx) * a + wx * b;
    precise float second = (1.0f - wx) * c + wx * d;
    precise float even = (1.0f - wy0) * first + wy0 * second;
    precise float odd = (1.0f - wy1) * first + wy1 * second;
    uint out0 = address(id.z, channel, y, x, params.outputH, params.outputW, params.outputOrder, params.outputPitch);
    uint out1 = address(id.z, channel, y + 1, x, params.outputH, params.outputW, params.outputOrder, params.outputPitch);
    outputData[out0] = storeHalf(even);
    outputData[out1] = storeHalf(odd);
}
