try:
    from typing import List, Optional, Union
except:
    pass

import flatbuffers
import numpy as np

# automatically generated via `bazel build //tensorflow/lite/python:schema_py`
# namespace: tflite


class LogicalOrOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLogicalOrOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalOrOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LogicalOrOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LogicalOrOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LogicalOrOptionsStart(builder):
    builder.StartObject(0)


def LogicalOrOptionsEnd(builder):
    return builder.EndObject()


class LogicalOrOptionsT(object):

    # LogicalOrOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalOrOptions = LogicalOrOptions()
        logicalOrOptions.Init(buf, pos)
        return cls.InitFromObj(logicalOrOptions)

    @classmethod
    def InitFromObj(cls, logicalOrOptions):
        x = LogicalOrOptionsT()
        x._UnPack(logicalOrOptions)
        return x

    # LogicalOrOptionsT
    def _UnPack(self, logicalOrOptions):
        if logicalOrOptions is None:
            return

    # LogicalOrOptionsT
    def Pack(self, builder):
        LogicalOrOptionsStart(builder)
        logicalOrOptions = LogicalOrOptionsEnd(builder)
        return logicalOrOptions


class ZerosLikeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsZerosLikeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ZerosLikeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ZerosLikeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ZerosLikeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ZerosLikeOptionsStart(builder):
    builder.StartObject(0)


def ZerosLikeOptionsEnd(builder):
    return builder.EndObject()


class ZerosLikeOptionsT(object):

    # ZerosLikeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        zerosLikeOptions = ZerosLikeOptions()
        zerosLikeOptions.Init(buf, pos)
        return cls.InitFromObj(zerosLikeOptions)

    @classmethod
    def InitFromObj(cls, zerosLikeOptions):
        x = ZerosLikeOptionsT()
        x._UnPack(zerosLikeOptions)
        return x

    # ZerosLikeOptionsT
    def _UnPack(self, zerosLikeOptions):
        if zerosLikeOptions is None:
            return

    # ZerosLikeOptionsT
    def Pack(self, builder):
        ZerosLikeOptionsStart(builder)
        zerosLikeOptions = ZerosLikeOptionsEnd(builder)
        return zerosLikeOptions


class EqualOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def EqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # EqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def EqualOptionsStart(builder):
    builder.StartObject(0)


def EqualOptionsEnd(builder):
    return builder.EndObject()


class EqualOptionsT(object):

    # EqualOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        equalOptions = EqualOptions()
        equalOptions.Init(buf, pos)
        return cls.InitFromObj(equalOptions)

    @classmethod
    def InitFromObj(cls, equalOptions):
        x = EqualOptionsT()
        x._UnPack(equalOptions)
        return x

    # EqualOptionsT
    def _UnPack(self, equalOptions):
        if equalOptions is None:
            return

    # EqualOptionsT
    def Pack(self, builder):
        EqualOptionsStart(builder)
        equalOptions = EqualOptionsEnd(builder)
        return equalOptions


class PowOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsPowOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PowOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def PowOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # PowOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def PowOptionsStart(builder):
    builder.StartObject(0)


def PowOptionsEnd(builder):
    return builder.EndObject()


class PowOptionsT(object):

    # PowOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        powOptions = PowOptions()
        powOptions.Init(buf, pos)
        return cls.InitFromObj(powOptions)

    @classmethod
    def InitFromObj(cls, powOptions):
        x = PowOptionsT()
        x._UnPack(powOptions)
        return x

    # PowOptionsT
    def _UnPack(self, powOptions):
        if powOptions is None:
            return

    # PowOptionsT
    def Pack(self, builder):
        PowOptionsStart(builder)
        powOptions = PowOptionsEnd(builder)
        return powOptions


class LessOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLessOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LessOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LessOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LessOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LessOptionsStart(builder):
    builder.StartObject(0)


def LessOptionsEnd(builder):
    return builder.EndObject()


class LessOptionsT(object):

    # LessOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lessOptions = LessOptions()
        lessOptions.Init(buf, pos)
        return cls.InitFromObj(lessOptions)

    @classmethod
    def InitFromObj(cls, lessOptions):
        x = LessOptionsT()
        x._UnPack(lessOptions)
        return x

    # LessOptionsT
    def _UnPack(self, lessOptions):
        if lessOptions is None:
            return

    # LessOptionsT
    def Pack(self, builder):
        LessOptionsStart(builder)
        lessOptions = LessOptionsEnd(builder)
        return lessOptions


class CombinerType(object):
    SUM = 0
    MEAN = 1
    SQRTN = 2


class AddOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsAddOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AddOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def AddOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # AddOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AddOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AddOptions
    def PotScaleInt16(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return True


def AddOptionsStart(builder):
    builder.StartObject(2)


def AddOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def AddOptionsAddPotScaleInt16(builder, potScaleInt16):
    builder.PrependBoolSlot(1, potScaleInt16, 1)


def AddOptionsEnd(builder):
    return builder.EndObject()


class AddOptionsT(object):

    # AddOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.potScaleInt16 = True  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        addOptions = AddOptions()
        addOptions.Init(buf, pos)
        return cls.InitFromObj(addOptions)

    @classmethod
    def InitFromObj(cls, addOptions):
        x = AddOptionsT()
        x._UnPack(addOptions)
        return x

    # AddOptionsT
    def _UnPack(self, addOptions):
        if addOptions is None:
            return
        self.fusedActivationFunction = addOptions.FusedActivationFunction()
        self.potScaleInt16 = addOptions.PotScaleInt16()

    # AddOptionsT
    def Pack(self, builder):
        AddOptionsStart(builder)
        AddOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        AddOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        addOptions = AddOptionsEnd(builder)
        return addOptions


class QuantizeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsQuantizeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = QuantizeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def QuantizeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # QuantizeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def QuantizeOptionsStart(builder):
    builder.StartObject(0)


def QuantizeOptionsEnd(builder):
    return builder.EndObject()


class QuantizeOptionsT(object):

    # QuantizeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quantizeOptions = QuantizeOptions()
        quantizeOptions.Init(buf, pos)
        return cls.InitFromObj(quantizeOptions)

    @classmethod
    def InitFromObj(cls, quantizeOptions):
        x = QuantizeOptionsT()
        x._UnPack(quantizeOptions)
        return x

    # QuantizeOptionsT
    def _UnPack(self, quantizeOptions):
        if quantizeOptions is None:
            return

    # QuantizeOptionsT
    def Pack(self, builder):
        QuantizeOptionsStart(builder)
        quantizeOptions = QuantizeOptionsEnd(builder)
        return quantizeOptions


class Pool2DOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsPool2DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Pool2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Pool2DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Pool2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Pool2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Pool2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Pool2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Pool2DOptions
    def FilterWidth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Pool2DOptions
    def FilterHeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Pool2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def Pool2DOptionsStart(builder):
    builder.StartObject(6)


def Pool2DOptionsAddPadding(builder, padding):
    builder.PrependInt8Slot(0, padding, 0)


def Pool2DOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(1, strideW, 0)


def Pool2DOptionsAddStrideH(builder, strideH):
    builder.PrependInt32Slot(2, strideH, 0)


def Pool2DOptionsAddFilterWidth(builder, filterWidth):
    builder.PrependInt32Slot(3, filterWidth, 0)


def Pool2DOptionsAddFilterHeight(builder, filterHeight):
    builder.PrependInt32Slot(4, filterHeight, 0)


def Pool2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(5, fusedActivationFunction, 0)


def Pool2DOptionsEnd(builder):
    return builder.EndObject()


class Pool2DOptionsT(object):

    # Pool2DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.filterWidth = 0  # type: int
        self.filterHeight = 0  # type: int
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        pool2DOptions = Pool2DOptions()
        pool2DOptions.Init(buf, pos)
        return cls.InitFromObj(pool2DOptions)

    @classmethod
    def InitFromObj(cls, pool2DOptions):
        x = Pool2DOptionsT()
        x._UnPack(pool2DOptions)
        return x

    # Pool2DOptionsT
    def _UnPack(self, pool2DOptions):
        if pool2DOptions is None:
            return
        self.padding = pool2DOptions.Padding()
        self.strideW = pool2DOptions.StrideW()
        self.strideH = pool2DOptions.StrideH()
        self.filterWidth = pool2DOptions.FilterWidth()
        self.filterHeight = pool2DOptions.FilterHeight()
        self.fusedActivationFunction = pool2DOptions.FusedActivationFunction()

    # Pool2DOptionsT
    def Pack(self, builder):
        Pool2DOptionsStart(builder)
        Pool2DOptionsAddPadding(builder, self.padding)
        Pool2DOptionsAddStrideW(builder, self.strideW)
        Pool2DOptionsAddStrideH(builder, self.strideH)
        Pool2DOptionsAddFilterWidth(builder, self.filterWidth)
        Pool2DOptionsAddFilterHeight(builder, self.filterHeight)
        Pool2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        pool2DOptions = Pool2DOptionsEnd(builder)
        return pool2DOptions


class SoftmaxOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSoftmaxOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SoftmaxOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SoftmaxOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SoftmaxOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SoftmaxOptions
    def Beta(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0


def SoftmaxOptionsStart(builder):
    builder.StartObject(1)


def SoftmaxOptionsAddBeta(builder, beta):
    builder.PrependFloat32Slot(0, beta, 0.0)


def SoftmaxOptionsEnd(builder):
    return builder.EndObject()


class SoftmaxOptionsT(object):

    # SoftmaxOptionsT
    def __init__(self):
        self.beta = 0.0  # type: float

    @classmethod
    def InitFromBuf(cls, buf, pos):
        softmaxOptions = SoftmaxOptions()
        softmaxOptions.Init(buf, pos)
        return cls.InitFromObj(softmaxOptions)

    @classmethod
    def InitFromObj(cls, softmaxOptions):
        x = SoftmaxOptionsT()
        x._UnPack(softmaxOptions)
        return x

    # SoftmaxOptionsT
    def _UnPack(self, softmaxOptions):
        if softmaxOptions is None:
            return
        self.beta = softmaxOptions.Beta()

    # SoftmaxOptionsT
    def Pack(self, builder):
        SoftmaxOptionsStart(builder)
        SoftmaxOptionsAddBeta(builder, self.beta)
        softmaxOptions = SoftmaxOptionsEnd(builder)
        return softmaxOptions


class ShapeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsShapeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ShapeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ShapeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ShapeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ShapeOptions
    def OutType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ShapeOptionsStart(builder):
    builder.StartObject(1)


def ShapeOptionsAddOutType(builder, outType):
    builder.PrependInt8Slot(0, outType, 0)


def ShapeOptionsEnd(builder):
    return builder.EndObject()


class ShapeOptionsT(object):

    # ShapeOptionsT
    def __init__(self):
        self.outType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        shapeOptions = ShapeOptions()
        shapeOptions.Init(buf, pos)
        return cls.InitFromObj(shapeOptions)

    @classmethod
    def InitFromObj(cls, shapeOptions):
        x = ShapeOptionsT()
        x._UnPack(shapeOptions)
        return x

    # ShapeOptionsT
    def _UnPack(self, shapeOptions):
        if shapeOptions is None:
            return
        self.outType = shapeOptions.OutType()

    # ShapeOptionsT
    def Pack(self, builder):
        ShapeOptionsStart(builder)
        ShapeOptionsAddOutType(builder, self.outType)
        shapeOptions = ShapeOptionsEnd(builder)
        return shapeOptions


class SparsityParameters(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSparsityParameters(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SparsityParameters()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SparsityParametersBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SparsityParameters
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SparsityParameters
    def TraversalOrder(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SparsityParameters
    def TraversalOrderAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SparsityParameters
    def TraversalOrderLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def TraversalOrderIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SparsityParameters
    def BlockMap(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SparsityParameters
    def BlockMapAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SparsityParameters
    def BlockMapLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def BlockMapIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SparsityParameters
    def DimMetadata(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = DimensionMetadata()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SparsityParameters
    def DimMetadataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def DimMetadataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0


def SparsityParametersStart(builder):
    builder.StartObject(3)


def SparsityParametersAddTraversalOrder(builder, traversalOrder):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(traversalOrder), 0
    )


def SparsityParametersStartTraversalOrderVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SparsityParametersAddBlockMap(builder, blockMap):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(blockMap), 0
    )


def SparsityParametersStartBlockMapVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SparsityParametersAddDimMetadata(builder, dimMetadata):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(dimMetadata), 0
    )


def SparsityParametersStartDimMetadataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SparsityParametersEnd(builder):
    return builder.EndObject()


class SparsityParametersT(object):

    # SparsityParametersT
    def __init__(self):
        self.traversalOrder = None  # type: List[int]
        self.blockMap = None  # type: List[int]
        self.dimMetadata = None  # type: List[DimensionMetadataT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparsityParameters = SparsityParameters()
        sparsityParameters.Init(buf, pos)
        return cls.InitFromObj(sparsityParameters)

    @classmethod
    def InitFromObj(cls, sparsityParameters):
        x = SparsityParametersT()
        x._UnPack(sparsityParameters)
        return x

    # SparsityParametersT
    def _UnPack(self, sparsityParameters):
        if sparsityParameters is None:
            return
        if not sparsityParameters.TraversalOrderIsNone():
            if np is None:
                self.traversalOrder = []
                for i in range(sparsityParameters.TraversalOrderLength()):
                    self.traversalOrder.append(sparsityParameters.TraversalOrder(i))
            else:
                self.traversalOrder = sparsityParameters.TraversalOrderAsNumpy()
        if not sparsityParameters.BlockMapIsNone():
            if np is None:
                self.blockMap = []
                for i in range(sparsityParameters.BlockMapLength()):
                    self.blockMap.append(sparsityParameters.BlockMap(i))
            else:
                self.blockMap = sparsityParameters.BlockMapAsNumpy()
        if not sparsityParameters.DimMetadataIsNone():
            self.dimMetadata = []
            for i in range(sparsityParameters.DimMetadataLength()):
                if sparsityParameters.DimMetadata(i) is None:
                    self.dimMetadata.append(None)
                else:
                    dimensionMetadata_ = DimensionMetadataT.InitFromObj(
                        sparsityParameters.DimMetadata(i)
                    )
                    self.dimMetadata.append(dimensionMetadata_)

    # SparsityParametersT
    def Pack(self, builder):
        if self.traversalOrder is not None:
            if np is not None and type(self.traversalOrder) is np.ndarray:
                traversalOrder = builder.CreateNumpyVector(self.traversalOrder)
            else:
                SparsityParametersStartTraversalOrderVector(
                    builder, len(self.traversalOrder)
                )
                for i in reversed(range(len(self.traversalOrder))):
                    builder.PrependInt32(self.traversalOrder[i])
                traversalOrder = builder.EndVector(len(self.traversalOrder))
        if self.blockMap is not None:
            if np is not None and type(self.blockMap) is np.ndarray:
                blockMap = builder.CreateNumpyVector(self.blockMap)
            else:
                SparsityParametersStartBlockMapVector(builder, len(self.blockMap))
                for i in reversed(range(len(self.blockMap))):
                    builder.PrependInt32(self.blockMap[i])
                blockMap = builder.EndVector(len(self.blockMap))
        if self.dimMetadata is not None:
            dimMetadatalist = []
            for i in range(len(self.dimMetadata)):
                dimMetadatalist.append(self.dimMetadata[i].Pack(builder))
            SparsityParametersStartDimMetadataVector(builder, len(self.dimMetadata))
            for i in reversed(range(len(self.dimMetadata))):
                builder.PrependUOffsetTRelative(dimMetadatalist[i])
            dimMetadata = builder.EndVector(len(self.dimMetadata))
        SparsityParametersStart(builder)
        if self.traversalOrder is not None:
            SparsityParametersAddTraversalOrder(builder, traversalOrder)
        if self.blockMap is not None:
            SparsityParametersAddBlockMap(builder, blockMap)
        if self.dimMetadata is not None:
            SparsityParametersAddDimMetadata(builder, dimMetadata)
        sparsityParameters = SparsityParametersEnd(builder)
        return sparsityParameters


class ReverseSequenceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsReverseSequenceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseSequenceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ReverseSequenceOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ReverseSequenceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReverseSequenceOptions
    def SeqDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReverseSequenceOptions
    def BatchDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def ReverseSequenceOptionsStart(builder):
    builder.StartObject(2)


def ReverseSequenceOptionsAddSeqDim(builder, seqDim):
    builder.PrependInt32Slot(0, seqDim, 0)


def ReverseSequenceOptionsAddBatchDim(builder, batchDim):
    builder.PrependInt32Slot(1, batchDim, 0)


def ReverseSequenceOptionsEnd(builder):
    return builder.EndObject()


class ReverseSequenceOptionsT(object):

    # ReverseSequenceOptionsT
    def __init__(self):
        self.seqDim = 0  # type: int
        self.batchDim = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reverseSequenceOptions = ReverseSequenceOptions()
        reverseSequenceOptions.Init(buf, pos)
        return cls.InitFromObj(reverseSequenceOptions)

    @classmethod
    def InitFromObj(cls, reverseSequenceOptions):
        x = ReverseSequenceOptionsT()
        x._UnPack(reverseSequenceOptions)
        return x

    # ReverseSequenceOptionsT
    def _UnPack(self, reverseSequenceOptions):
        if reverseSequenceOptions is None:
            return
        self.seqDim = reverseSequenceOptions.SeqDim()
        self.batchDim = reverseSequenceOptions.BatchDim()

    # ReverseSequenceOptionsT
    def Pack(self, builder):
        ReverseSequenceOptionsStart(builder)
        ReverseSequenceOptionsAddSeqDim(builder, self.seqDim)
        ReverseSequenceOptionsAddBatchDim(builder, self.batchDim)
        reverseSequenceOptions = ReverseSequenceOptionsEnd(builder)
        return reverseSequenceOptions


class LeakyReluOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLeakyReluOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LeakyReluOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LeakyReluOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LeakyReluOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LeakyReluOptions
    def Alpha(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0


def LeakyReluOptionsStart(builder):
    builder.StartObject(1)


def LeakyReluOptionsAddAlpha(builder, alpha):
    builder.PrependFloat32Slot(0, alpha, 0.0)


def LeakyReluOptionsEnd(builder):
    return builder.EndObject()


class LeakyReluOptionsT(object):

    # LeakyReluOptionsT
    def __init__(self):
        self.alpha = 0.0  # type: float

    @classmethod
    def InitFromBuf(cls, buf, pos):
        leakyReluOptions = LeakyReluOptions()
        leakyReluOptions.Init(buf, pos)
        return cls.InitFromObj(leakyReluOptions)

    @classmethod
    def InitFromObj(cls, leakyReluOptions):
        x = LeakyReluOptionsT()
        x._UnPack(leakyReluOptions)
        return x

    # LeakyReluOptionsT
    def _UnPack(self, leakyReluOptions):
        if leakyReluOptions is None:
            return
        self.alpha = leakyReluOptions.Alpha()

    # LeakyReluOptionsT
    def Pack(self, builder):
        LeakyReluOptionsStart(builder)
        LeakyReluOptionsAddAlpha(builder, self.alpha)
        leakyReluOptions = LeakyReluOptionsEnd(builder)
        return leakyReluOptions


class LSHProjectionOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLSHProjectionOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LSHProjectionOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LSHProjectionOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LSHProjectionOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LSHProjectionOptions
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def LSHProjectionOptionsStart(builder):
    builder.StartObject(1)


def LSHProjectionOptionsAddType(builder, type):
    builder.PrependInt8Slot(0, type, 0)


def LSHProjectionOptionsEnd(builder):
    return builder.EndObject()


class LSHProjectionOptionsT(object):

    # LSHProjectionOptionsT
    def __init__(self):
        self.type = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lSHProjectionOptions = LSHProjectionOptions()
        lSHProjectionOptions.Init(buf, pos)
        return cls.InitFromObj(lSHProjectionOptions)

    @classmethod
    def InitFromObj(cls, lSHProjectionOptions):
        x = LSHProjectionOptionsT()
        x._UnPack(lSHProjectionOptions)
        return x

    # LSHProjectionOptionsT
    def _UnPack(self, lSHProjectionOptions):
        if lSHProjectionOptions is None:
            return
        self.type = lSHProjectionOptions.Type()

    # LSHProjectionOptionsT
    def Pack(self, builder):
        LSHProjectionOptionsStart(builder)
        LSHProjectionOptionsAddType(builder, self.type)
        lSHProjectionOptions = LSHProjectionOptionsEnd(builder)
        return lSHProjectionOptions


class Metadata(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMetadata(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Metadata()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MetadataBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Metadata
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Metadata
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Metadata
    def Buffer(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Uint32Flags, o + self._tab.Pos
            )
        return 0


def MetadataStart(builder):
    builder.StartObject(2)


def MetadataAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0
    )


def MetadataAddBuffer(builder, buffer):
    builder.PrependUint32Slot(1, buffer, 0)


def MetadataEnd(builder):
    return builder.EndObject()


class MetadataT(object):

    # MetadataT
    def __init__(self):
        self.name = None  # type: str
        self.buffer = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        metadata = Metadata()
        metadata.Init(buf, pos)
        return cls.InitFromObj(metadata)

    @classmethod
    def InitFromObj(cls, metadata):
        x = MetadataT()
        x._UnPack(metadata)
        return x

    # MetadataT
    def _UnPack(self, metadata):
        if metadata is None:
            return
        self.name = metadata.Name()
        self.buffer = metadata.Buffer()

    # MetadataT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        MetadataStart(builder)
        if self.name is not None:
            MetadataAddName(builder, name)
        MetadataAddBuffer(builder, self.buffer)
        metadata = MetadataEnd(builder)
        return metadata


class MatrixDiagOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMatrixDiagOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixDiagOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MatrixDiagOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # MatrixDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def MatrixDiagOptionsStart(builder):
    builder.StartObject(0)


def MatrixDiagOptionsEnd(builder):
    return builder.EndObject()


class MatrixDiagOptionsT(object):

    # MatrixDiagOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        matrixDiagOptions = MatrixDiagOptions()
        matrixDiagOptions.Init(buf, pos)
        return cls.InitFromObj(matrixDiagOptions)

    @classmethod
    def InitFromObj(cls, matrixDiagOptions):
        x = MatrixDiagOptionsT()
        x._UnPack(matrixDiagOptions)
        return x

    # MatrixDiagOptionsT
    def _UnPack(self, matrixDiagOptions):
        if matrixDiagOptions is None:
            return

    # MatrixDiagOptionsT
    def Pack(self, builder):
        MatrixDiagOptionsStart(builder)
        matrixDiagOptions = MatrixDiagOptionsEnd(builder)
        return matrixDiagOptions


class LessEqualOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLessEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LessEqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LessEqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LessEqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LessEqualOptionsStart(builder):
    builder.StartObject(0)


def LessEqualOptionsEnd(builder):
    return builder.EndObject()


class LessEqualOptionsT(object):

    # LessEqualOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lessEqualOptions = LessEqualOptions()
        lessEqualOptions.Init(buf, pos)
        return cls.InitFromObj(lessEqualOptions)

    @classmethod
    def InitFromObj(cls, lessEqualOptions):
        x = LessEqualOptionsT()
        x._UnPack(lessEqualOptions)
        return x

    # LessEqualOptionsT
    def _UnPack(self, lessEqualOptions):
        if lessEqualOptions is None:
            return

    # LessEqualOptionsT
    def Pack(self, builder):
        LessEqualOptionsStart(builder)
        lessEqualOptions = LessEqualOptionsEnd(builder)
        return lessEqualOptions


class ReverseV2Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsReverseV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ReverseV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ReverseV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ReverseV2OptionsStart(builder):
    builder.StartObject(0)


def ReverseV2OptionsEnd(builder):
    return builder.EndObject()


class ReverseV2OptionsT(object):

    # ReverseV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reverseV2Options = ReverseV2Options()
        reverseV2Options.Init(buf, pos)
        return cls.InitFromObj(reverseV2Options)

    @classmethod
    def InitFromObj(cls, reverseV2Options):
        x = ReverseV2OptionsT()
        x._UnPack(reverseV2Options)
        return x

    # ReverseV2OptionsT
    def _UnPack(self, reverseV2Options):
        if reverseV2Options is None:
            return

    # ReverseV2OptionsT
    def Pack(self, builder):
        ReverseV2OptionsStart(builder)
        reverseV2Options = ReverseV2OptionsEnd(builder)
        return reverseV2Options


class RNNOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsRNNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def RNNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # RNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # RNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def RNNOptionsStart(builder):
    builder.StartObject(2)


def RNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def RNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(1, asymmetricQuantizeInputs, 0)


def RNNOptionsEnd(builder):
    return builder.EndObject()


class RNNOptionsT(object):

    # RNNOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rNNOptions = RNNOptions()
        rNNOptions.Init(buf, pos)
        return cls.InitFromObj(rNNOptions)

    @classmethod
    def InitFromObj(cls, rNNOptions):
        x = RNNOptionsT()
        x._UnPack(rNNOptions)
        return x

    # RNNOptionsT
    def _UnPack(self, rNNOptions):
        if rNNOptions is None:
            return
        self.fusedActivationFunction = rNNOptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = rNNOptions.AsymmetricQuantizeInputs()

    # RNNOptionsT
    def Pack(self, builder):
        RNNOptionsStart(builder)
        RNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        RNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        rNNOptions = RNNOptionsEnd(builder)
        return rNNOptions


class DimensionMetadata(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDimensionMetadata(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DimensionMetadata()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DimensionMetadataBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DimensionMetadata
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DimensionMetadata
    def Format(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def DenseSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegmentsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegments(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            from flatbuffers.table import Table

            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # DimensionMetadata
    def ArrayIndicesType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArrayIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            from flatbuffers.table import Table

            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None


def DimensionMetadataStart(builder):
    builder.StartObject(6)


def DimensionMetadataAddFormat(builder, format):
    builder.PrependInt8Slot(0, format, 0)


def DimensionMetadataAddDenseSize(builder, denseSize):
    builder.PrependInt32Slot(1, denseSize, 0)


def DimensionMetadataAddArraySegmentsType(builder, arraySegmentsType):
    builder.PrependUint8Slot(2, arraySegmentsType, 0)


def DimensionMetadataAddArraySegments(builder, arraySegments):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(arraySegments), 0
    )


def DimensionMetadataAddArrayIndicesType(builder, arrayIndicesType):
    builder.PrependUint8Slot(4, arrayIndicesType, 0)


def DimensionMetadataAddArrayIndices(builder, arrayIndices):
    builder.PrependUOffsetTRelativeSlot(
        5, flatbuffers.number_types.UOffsetTFlags.py_type(arrayIndices), 0
    )


def DimensionMetadataEnd(builder):
    return builder.EndObject()


class DimensionMetadataT(object):

    # DimensionMetadataT
    def __init__(self):
        self.format = 0  # type: int
        self.denseSize = 0  # type: int
        self.arraySegmentsType = 0  # type: int
        self.arraySegments = (
            None
        )  # type: Union[None, Int32VectorT, Uint16VectorT, Uint8VectorT]
        self.arrayIndicesType = 0  # type: int
        self.arrayIndices = (
            None
        )  # type: Union[None, Int32VectorT, Uint16VectorT, Uint8VectorT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dimensionMetadata = DimensionMetadata()
        dimensionMetadata.Init(buf, pos)
        return cls.InitFromObj(dimensionMetadata)

    @classmethod
    def InitFromObj(cls, dimensionMetadata):
        x = DimensionMetadataT()
        x._UnPack(dimensionMetadata)
        return x

    # DimensionMetadataT
    def _UnPack(self, dimensionMetadata):
        if dimensionMetadata is None:
            return
        self.format = dimensionMetadata.Format()
        self.denseSize = dimensionMetadata.DenseSize()
        self.arraySegmentsType = dimensionMetadata.ArraySegmentsType()
        self.arraySegments = SparseIndexVectorCreator(
            self.arraySegmentsType, dimensionMetadata.ArraySegments()
        )
        self.arrayIndicesType = dimensionMetadata.ArrayIndicesType()
        self.arrayIndices = SparseIndexVectorCreator(
            self.arrayIndicesType, dimensionMetadata.ArrayIndices()
        )

    # DimensionMetadataT
    def Pack(self, builder):
        if self.arraySegments is not None:
            arraySegments = self.arraySegments.Pack(builder)
        if self.arrayIndices is not None:
            arrayIndices = self.arrayIndices.Pack(builder)
        DimensionMetadataStart(builder)
        DimensionMetadataAddFormat(builder, self.format)
        DimensionMetadataAddDenseSize(builder, self.denseSize)
        DimensionMetadataAddArraySegmentsType(builder, self.arraySegmentsType)
        if self.arraySegments is not None:
            DimensionMetadataAddArraySegments(builder, arraySegments)
        DimensionMetadataAddArrayIndicesType(builder, self.arrayIndicesType)
        if self.arrayIndices is not None:
            DimensionMetadataAddArrayIndices(builder, arrayIndices)
        dimensionMetadata = DimensionMetadataEnd(builder)
        return dimensionMetadata


class ActivationFunctionType(object):
    NONE = 0
    RELU = 1
    RELU_N1_TO_1 = 2
    RELU6 = 3
    TANH = 4
    SIGN_BIT = 5


class Int32Vector(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsInt32Vector(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Int32Vector()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Int32VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Int32Vector
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Int32Vector
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Int32Vector
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Int32Vector
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Int32Vector
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def Int32VectorStart(builder):
    builder.StartObject(1)


def Int32VectorAddValues(builder, values):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0
    )


def Int32VectorStartValuesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def Int32VectorEnd(builder):
    return builder.EndObject()


class Int32VectorT(object):

    # Int32VectorT
    def __init__(self):
        self.values = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        int32Vector = Int32Vector()
        int32Vector.Init(buf, pos)
        return cls.InitFromObj(int32Vector)

    @classmethod
    def InitFromObj(cls, int32Vector):
        x = Int32VectorT()
        x._UnPack(int32Vector)
        return x

    # Int32VectorT
    def _UnPack(self, int32Vector):
        if int32Vector is None:
            return
        if not int32Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(int32Vector.ValuesLength()):
                    self.values.append(int32Vector.Values(i))
            else:
                self.values = int32Vector.ValuesAsNumpy()

    # Int32VectorT
    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Int32VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependInt32(self.values[i])
                values = builder.EndVector(len(self.values))
        Int32VectorStart(builder)
        if self.values is not None:
            Int32VectorAddValues(builder, values)
        int32Vector = Int32VectorEnd(builder)
        return int32Vector


class NegOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsNegOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NegOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NegOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # NegOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NegOptionsStart(builder):
    builder.StartObject(0)


def NegOptionsEnd(builder):
    return builder.EndObject()


class NegOptionsT(object):

    # NegOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        negOptions = NegOptions()
        negOptions.Init(buf, pos)
        return cls.InitFromObj(negOptions)

    @classmethod
    def InitFromObj(cls, negOptions):
        x = NegOptionsT()
        x._UnPack(negOptions)
        return x

    # NegOptionsT
    def _UnPack(self, negOptions):
        if negOptions is None:
            return

    # NegOptionsT
    def Pack(self, builder):
        NegOptionsStart(builder)
        negOptions = NegOptionsEnd(builder)
        return negOptions


class FloorDivOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsFloorDivOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloorDivOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FloorDivOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # FloorDivOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def FloorDivOptionsStart(builder):
    builder.StartObject(0)


def FloorDivOptionsEnd(builder):
    return builder.EndObject()


class FloorDivOptionsT(object):

    # FloorDivOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        floorDivOptions = FloorDivOptions()
        floorDivOptions.Init(buf, pos)
        return cls.InitFromObj(floorDivOptions)

    @classmethod
    def InitFromObj(cls, floorDivOptions):
        x = FloorDivOptionsT()
        x._UnPack(floorDivOptions)
        return x

    # FloorDivOptionsT
    def _UnPack(self, floorDivOptions):
        if floorDivOptions is None:
            return

    # FloorDivOptionsT
    def Pack(self, builder):
        FloorDivOptionsStart(builder)
        floorDivOptions = FloorDivOptionsEnd(builder)
        return floorDivOptions


class LogSoftmaxOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLogSoftmaxOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogSoftmaxOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LogSoftmaxOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LogSoftmaxOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LogSoftmaxOptionsStart(builder):
    builder.StartObject(0)


def LogSoftmaxOptionsEnd(builder):
    return builder.EndObject()


class LogSoftmaxOptionsT(object):

    # LogSoftmaxOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logSoftmaxOptions = LogSoftmaxOptions()
        logSoftmaxOptions.Init(buf, pos)
        return cls.InitFromObj(logSoftmaxOptions)

    @classmethod
    def InitFromObj(cls, logSoftmaxOptions):
        x = LogSoftmaxOptionsT()
        x._UnPack(logSoftmaxOptions)
        return x

    # LogSoftmaxOptionsT
    def _UnPack(self, logSoftmaxOptions):
        if logSoftmaxOptions is None:
            return

    # LogSoftmaxOptionsT
    def Pack(self, builder):
        LogSoftmaxOptionsStart(builder)
        logSoftmaxOptions = LogSoftmaxOptionsEnd(builder)
        return logSoftmaxOptions


class MirrorPadMode(object):
    REFLECT = 0
    SYMMETRIC = 1


class SelectOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSelectOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SelectOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SelectOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SelectOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SelectOptionsStart(builder):
    builder.StartObject(0)


def SelectOptionsEnd(builder):
    return builder.EndObject()


class SelectOptionsT(object):

    # SelectOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        selectOptions = SelectOptions()
        selectOptions.Init(buf, pos)
        return cls.InitFromObj(selectOptions)

    @classmethod
    def InitFromObj(cls, selectOptions):
        x = SelectOptionsT()
        x._UnPack(selectOptions)
        return x

    # SelectOptionsT
    def _UnPack(self, selectOptions):
        if selectOptions is None:
            return

    # SelectOptionsT
    def Pack(self, builder):
        SelectOptionsStart(builder)
        selectOptions = SelectOptionsEnd(builder)
        return selectOptions


class UniqueOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsUniqueOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UniqueOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def UniqueOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # UniqueOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UniqueOptions
    def IdxOutType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 2


def UniqueOptionsStart(builder):
    builder.StartObject(1)


def UniqueOptionsAddIdxOutType(builder, idxOutType):
    builder.PrependInt8Slot(0, idxOutType, 2)


def UniqueOptionsEnd(builder):
    return builder.EndObject()


class UniqueOptionsT(object):

    # UniqueOptionsT
    def __init__(self):
        self.idxOutType = 2  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uniqueOptions = UniqueOptions()
        uniqueOptions.Init(buf, pos)
        return cls.InitFromObj(uniqueOptions)

    @classmethod
    def InitFromObj(cls, uniqueOptions):
        x = UniqueOptionsT()
        x._UnPack(uniqueOptions)
        return x

    # UniqueOptionsT
    def _UnPack(self, uniqueOptions):
        if uniqueOptions is None:
            return
        self.idxOutType = uniqueOptions.IdxOutType()

    # UniqueOptionsT
    def Pack(self, builder):
        UniqueOptionsStart(builder)
        UniqueOptionsAddIdxOutType(builder, self.idxOutType)
        uniqueOptions = UniqueOptionsEnd(builder)
        return uniqueOptions


class GreaterOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsGreaterOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GreaterOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GreaterOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # GreaterOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def GreaterOptionsStart(builder):
    builder.StartObject(0)


def GreaterOptionsEnd(builder):
    return builder.EndObject()


class GreaterOptionsT(object):

    # GreaterOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        greaterOptions = GreaterOptions()
        greaterOptions.Init(buf, pos)
        return cls.InitFromObj(greaterOptions)

    @classmethod
    def InitFromObj(cls, greaterOptions):
        x = GreaterOptionsT()
        x._UnPack(greaterOptions)
        return x

    # GreaterOptionsT
    def _UnPack(self, greaterOptions):
        if greaterOptions is None:
            return

    # GreaterOptionsT
    def Pack(self, builder):
        GreaterOptionsStart(builder)
        greaterOptions = GreaterOptionsEnd(builder)
        return greaterOptions


class Uint8Vector(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsUint8Vector(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Uint8Vector()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Uint8VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Uint8Vector
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Uint8Vector
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Uint8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # Uint8Vector
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Uint8Vector
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Uint8Vector
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def Uint8VectorStart(builder):
    builder.StartObject(1)


def Uint8VectorAddValues(builder, values):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0
    )


def Uint8VectorStartValuesVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def Uint8VectorEnd(builder):
    return builder.EndObject()


class Uint8VectorT(object):

    # Uint8VectorT
    def __init__(self):
        self.values = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uint8Vector = Uint8Vector()
        uint8Vector.Init(buf, pos)
        return cls.InitFromObj(uint8Vector)

    @classmethod
    def InitFromObj(cls, uint8Vector):
        x = Uint8VectorT()
        x._UnPack(uint8Vector)
        return x

    # Uint8VectorT
    def _UnPack(self, uint8Vector):
        if uint8Vector is None:
            return
        if not uint8Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(uint8Vector.ValuesLength()):
                    self.values.append(uint8Vector.Values(i))
            else:
                self.values = uint8Vector.ValuesAsNumpy()

    # Uint8VectorT
    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Uint8VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint8(self.values[i])
                values = builder.EndVector(len(self.values))
        Uint8VectorStart(builder)
        if self.values is not None:
            Uint8VectorAddValues(builder, values)
        uint8Vector = Uint8VectorEnd(builder)
        return uint8Vector


class RankOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsRankOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RankOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def RankOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # RankOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def RankOptionsStart(builder):
    builder.StartObject(0)


def RankOptionsEnd(builder):
    return builder.EndObject()


class RankOptionsT(object):

    # RankOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rankOptions = RankOptions()
        rankOptions.Init(buf, pos)
        return cls.InitFromObj(rankOptions)

    @classmethod
    def InitFromObj(cls, rankOptions):
        x = RankOptionsT()
        x._UnPack(rankOptions)
        return x

    # RankOptionsT
    def _UnPack(self, rankOptions):
        if rankOptions is None:
            return

    # RankOptionsT
    def Pack(self, builder):
        RankOptionsStart(builder)
        rankOptions = RankOptionsEnd(builder)
        return rankOptions


class ResizeBilinearOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsResizeBilinearOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeBilinearOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ResizeBilinearOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ResizeBilinearOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ResizeBilinearOptions
    def AlignCorners(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # ResizeBilinearOptions
    def HalfPixelCenters(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def ResizeBilinearOptionsStart(builder):
    builder.StartObject(4)


def ResizeBilinearOptionsAddAlignCorners(builder, alignCorners):
    builder.PrependBoolSlot(2, alignCorners, 0)


def ResizeBilinearOptionsAddHalfPixelCenters(builder, halfPixelCenters):
    builder.PrependBoolSlot(3, halfPixelCenters, 0)


def ResizeBilinearOptionsEnd(builder):
    return builder.EndObject()


class ResizeBilinearOptionsT(object):

    # ResizeBilinearOptionsT
    def __init__(self):
        self.alignCorners = False  # type: bool
        self.halfPixelCenters = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        resizeBilinearOptions = ResizeBilinearOptions()
        resizeBilinearOptions.Init(buf, pos)
        return cls.InitFromObj(resizeBilinearOptions)

    @classmethod
    def InitFromObj(cls, resizeBilinearOptions):
        x = ResizeBilinearOptionsT()
        x._UnPack(resizeBilinearOptions)
        return x

    # ResizeBilinearOptionsT
    def _UnPack(self, resizeBilinearOptions):
        if resizeBilinearOptions is None:
            return
        self.alignCorners = resizeBilinearOptions.AlignCorners()
        self.halfPixelCenters = resizeBilinearOptions.HalfPixelCenters()

    # ResizeBilinearOptionsT
    def Pack(self, builder):
        ResizeBilinearOptionsStart(builder)
        ResizeBilinearOptionsAddAlignCorners(builder, self.alignCorners)
        ResizeBilinearOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        resizeBilinearOptions = ResizeBilinearOptionsEnd(builder)
        return resizeBilinearOptions


class DivOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDivOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DivOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DivOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DivOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DivOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def DivOptionsStart(builder):
    builder.StartObject(1)


def DivOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def DivOptionsEnd(builder):
    return builder.EndObject()


class DivOptionsT(object):

    # DivOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        divOptions = DivOptions()
        divOptions.Init(buf, pos)
        return cls.InitFromObj(divOptions)

    @classmethod
    def InitFromObj(cls, divOptions):
        x = DivOptionsT()
        x._UnPack(divOptions)
        return x

    # DivOptionsT
    def _UnPack(self, divOptions):
        if divOptions is None:
            return
        self.fusedActivationFunction = divOptions.FusedActivationFunction()

    # DivOptionsT
    def Pack(self, builder):
        DivOptionsStart(builder)
        DivOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        divOptions = DivOptionsEnd(builder)
        return divOptions


class ReshapeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsReshapeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReshapeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ReshapeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ReshapeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReshapeOptions
    def NewShape(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # ReshapeOptions
    def NewShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ReshapeOptions
    def NewShapeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReshapeOptions
    def NewShapeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def ReshapeOptionsStart(builder):
    builder.StartObject(1)


def ReshapeOptionsAddNewShape(builder, newShape):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(newShape), 0
    )


def ReshapeOptionsStartNewShapeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ReshapeOptionsEnd(builder):
    return builder.EndObject()


class ReshapeOptionsT(object):

    # ReshapeOptionsT
    def __init__(self):
        self.newShape = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reshapeOptions = ReshapeOptions()
        reshapeOptions.Init(buf, pos)
        return cls.InitFromObj(reshapeOptions)

    @classmethod
    def InitFromObj(cls, reshapeOptions):
        x = ReshapeOptionsT()
        x._UnPack(reshapeOptions)
        return x

    # ReshapeOptionsT
    def _UnPack(self, reshapeOptions):
        if reshapeOptions is None:
            return
        if not reshapeOptions.NewShapeIsNone():
            if np is None:
                self.newShape = []
                for i in range(reshapeOptions.NewShapeLength()):
                    self.newShape.append(reshapeOptions.NewShape(i))
            else:
                self.newShape = reshapeOptions.NewShapeAsNumpy()

    # ReshapeOptionsT
    def Pack(self, builder):
        if self.newShape is not None:
            if np is not None and type(self.newShape) is np.ndarray:
                newShape = builder.CreateNumpyVector(self.newShape)
            else:
                ReshapeOptionsStartNewShapeVector(builder, len(self.newShape))
                for i in reversed(range(len(self.newShape))):
                    builder.PrependInt32(self.newShape[i])
                newShape = builder.EndVector(len(self.newShape))
        ReshapeOptionsStart(builder)
        if self.newShape is not None:
            ReshapeOptionsAddNewShape(builder, newShape)
        reshapeOptions = ReshapeOptionsEnd(builder)
        return reshapeOptions


class SelectV2Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSelectV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SelectV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SelectV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SelectV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SelectV2OptionsStart(builder):
    builder.StartObject(0)


def SelectV2OptionsEnd(builder):
    return builder.EndObject()


class SelectV2OptionsT(object):

    # SelectV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        selectV2Options = SelectV2Options()
        selectV2Options.Init(buf, pos)
        return cls.InitFromObj(selectV2Options)

    @classmethod
    def InitFromObj(cls, selectV2Options):
        x = SelectV2OptionsT()
        x._UnPack(selectV2Options)
        return x

    # SelectV2OptionsT
    def _UnPack(self, selectV2Options):
        if selectV2Options is None:
            return

    # SelectV2OptionsT
    def Pack(self, builder):
        SelectV2OptionsStart(builder)
        selectV2Options = SelectV2OptionsEnd(builder)
        return selectV2Options


class TopKV2Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsTopKV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TopKV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def TopKV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # TopKV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def TopKV2OptionsStart(builder):
    builder.StartObject(0)


def TopKV2OptionsEnd(builder):
    return builder.EndObject()


class TopKV2OptionsT(object):

    # TopKV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        topKV2Options = TopKV2Options()
        topKV2Options.Init(buf, pos)
        return cls.InitFromObj(topKV2Options)

    @classmethod
    def InitFromObj(cls, topKV2Options):
        x = TopKV2OptionsT()
        x._UnPack(topKV2Options)
        return x

    # TopKV2OptionsT
    def _UnPack(self, topKV2Options):
        if topKV2Options is None:
            return

    # TopKV2OptionsT
    def Pack(self, builder):
        TopKV2OptionsStart(builder)
        topKV2Options = TopKV2OptionsEnd(builder)
        return topKV2Options


class TileOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsTileOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TileOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def TileOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # TileOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def TileOptionsStart(builder):
    builder.StartObject(0)


def TileOptionsEnd(builder):
    return builder.EndObject()


class TileOptionsT(object):

    # TileOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tileOptions = TileOptions()
        tileOptions.Init(buf, pos)
        return cls.InitFromObj(tileOptions)

    @classmethod
    def InitFromObj(cls, tileOptions):
        x = TileOptionsT()
        x._UnPack(tileOptions)
        return x

    # TileOptionsT
    def _UnPack(self, tileOptions):
        if tileOptions is None:
            return

    # TileOptionsT
    def Pack(self, builder):
        TileOptionsStart(builder)
        tileOptions = TileOptionsEnd(builder)
        return tileOptions


class NotEqualOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsNotEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NotEqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NotEqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # NotEqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NotEqualOptionsStart(builder):
    builder.StartObject(0)


def NotEqualOptionsEnd(builder):
    return builder.EndObject()


class NotEqualOptionsT(object):

    # NotEqualOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        notEqualOptions = NotEqualOptions()
        notEqualOptions.Init(buf, pos)
        return cls.InitFromObj(notEqualOptions)

    @classmethod
    def InitFromObj(cls, notEqualOptions):
        x = NotEqualOptionsT()
        x._UnPack(notEqualOptions)
        return x

    # NotEqualOptionsT
    def _UnPack(self, notEqualOptions):
        if notEqualOptions is None:
            return

    # NotEqualOptionsT
    def Pack(self, builder):
        NotEqualOptionsStart(builder)
        notEqualOptions = NotEqualOptionsEnd(builder)
        return notEqualOptions


class AbsOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsAbsOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AbsOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def AbsOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # AbsOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def AbsOptionsStart(builder):
    builder.StartObject(0)


def AbsOptionsEnd(builder):
    return builder.EndObject()


class AbsOptionsT(object):

    # AbsOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        absOptions = AbsOptions()
        absOptions.Init(buf, pos)
        return cls.InitFromObj(absOptions)

    @classmethod
    def InitFromObj(cls, absOptions):
        x = AbsOptionsT()
        x._UnPack(absOptions)
        return x

    # AbsOptionsT
    def _UnPack(self, absOptions):
        if absOptions is None:
            return

    # AbsOptionsT
    def Pack(self, builder):
        AbsOptionsStart(builder)
        absOptions = AbsOptionsEnd(builder)
        return absOptions


class CustomOptionsFormat(object):
    FLEXBUFFERS = 0


class CustomQuantization(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsCustomQuantization(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CustomQuantization()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CustomQuantizationBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # CustomQuantization
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CustomQuantization
    def Custom(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Uint8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # CustomQuantization
    def CustomAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # CustomQuantization
    def CustomLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CustomQuantization
    def CustomIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def CustomQuantizationStart(builder):
    builder.StartObject(1)


def CustomQuantizationAddCustom(builder, custom):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(custom), 0
    )


def CustomQuantizationStartCustomVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def CustomQuantizationEnd(builder):
    return builder.EndObject()


class CustomQuantizationT(object):

    # CustomQuantizationT
    def __init__(self):
        self.custom = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        customQuantization = CustomQuantization()
        customQuantization.Init(buf, pos)
        return cls.InitFromObj(customQuantization)

    @classmethod
    def InitFromObj(cls, customQuantization):
        x = CustomQuantizationT()
        x._UnPack(customQuantization)
        return x

    # CustomQuantizationT
    def _UnPack(self, customQuantization):
        if customQuantization is None:
            return
        if not customQuantization.CustomIsNone():
            if np is None:
                self.custom = []
                for i in range(customQuantization.CustomLength()):
                    self.custom.append(customQuantization.Custom(i))
            else:
                self.custom = customQuantization.CustomAsNumpy()

    # CustomQuantizationT
    def Pack(self, builder):
        if self.custom is not None:
            if np is not None and type(self.custom) is np.ndarray:
                custom = builder.CreateNumpyVector(self.custom)
            else:
                CustomQuantizationStartCustomVector(builder, len(self.custom))
                for i in reversed(range(len(self.custom))):
                    builder.PrependUint8(self.custom[i])
                custom = builder.EndVector(len(self.custom))
        CustomQuantizationStart(builder)
        if self.custom is not None:
            CustomQuantizationAddCustom(builder, custom)
        customQuantization = CustomQuantizationEnd(builder)
        return customQuantization


class LSTMOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLSTMOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LSTMOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # LSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # LSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # LSTMOptions
    def KernelType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # LSTMOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def LSTMOptionsStart(builder):
    builder.StartObject(5)


def LSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def LSTMOptionsAddCellClip(builder, cellClip):
    builder.PrependFloat32Slot(1, cellClip, 0.0)


def LSTMOptionsAddProjClip(builder, projClip):
    builder.PrependFloat32Slot(2, projClip, 0.0)


def LSTMOptionsAddKernelType(builder, kernelType):
    builder.PrependInt8Slot(3, kernelType, 0)


def LSTMOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(4, asymmetricQuantizeInputs, 0)


def LSTMOptionsEnd(builder):
    return builder.EndObject()


class LSTMOptionsT(object):

    # LSTMOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.cellClip = 0.0  # type: float
        self.projClip = 0.0  # type: float
        self.kernelType = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lSTMOptions = LSTMOptions()
        lSTMOptions.Init(buf, pos)
        return cls.InitFromObj(lSTMOptions)

    @classmethod
    def InitFromObj(cls, lSTMOptions):
        x = LSTMOptionsT()
        x._UnPack(lSTMOptions)
        return x

    # LSTMOptionsT
    def _UnPack(self, lSTMOptions):
        if lSTMOptions is None:
            return
        self.fusedActivationFunction = lSTMOptions.FusedActivationFunction()
        self.cellClip = lSTMOptions.CellClip()
        self.projClip = lSTMOptions.ProjClip()
        self.kernelType = lSTMOptions.KernelType()
        self.asymmetricQuantizeInputs = lSTMOptions.AsymmetricQuantizeInputs()

    # LSTMOptionsT
    def Pack(self, builder):
        LSTMOptionsStart(builder)
        LSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        LSTMOptionsAddCellClip(builder, self.cellClip)
        LSTMOptionsAddProjClip(builder, self.projClip)
        LSTMOptionsAddKernelType(builder, self.kernelType)
        LSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        lSTMOptions = LSTMOptionsEnd(builder)
        return lSTMOptions


class SliceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSliceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SliceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SliceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SliceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SliceOptionsStart(builder):
    builder.StartObject(0)


def SliceOptionsEnd(builder):
    return builder.EndObject()


class SliceOptionsT(object):

    # SliceOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sliceOptions = SliceOptions()
        sliceOptions.Init(buf, pos)
        return cls.InitFromObj(sliceOptions)

    @classmethod
    def InitFromObj(cls, sliceOptions):
        x = SliceOptionsT()
        x._UnPack(sliceOptions)
        return x

    # SliceOptionsT
    def _UnPack(self, sliceOptions):
        if sliceOptions is None:
            return

    # SliceOptionsT
    def Pack(self, builder):
        SliceOptionsStart(builder)
        sliceOptions = SliceOptionsEnd(builder)
        return sliceOptions


class ArgMinOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsArgMinOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArgMinOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ArgMinOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ArgMinOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArgMinOptions
    def OutputType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ArgMinOptionsStart(builder):
    builder.StartObject(1)


def ArgMinOptionsAddOutputType(builder, outputType):
    builder.PrependInt8Slot(0, outputType, 0)


def ArgMinOptionsEnd(builder):
    return builder.EndObject()


class ArgMinOptionsT(object):

    # ArgMinOptionsT
    def __init__(self):
        self.outputType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        argMinOptions = ArgMinOptions()
        argMinOptions.Init(buf, pos)
        return cls.InitFromObj(argMinOptions)

    @classmethod
    def InitFromObj(cls, argMinOptions):
        x = ArgMinOptionsT()
        x._UnPack(argMinOptions)
        return x

    # ArgMinOptionsT
    def _UnPack(self, argMinOptions):
        if argMinOptions is None:
            return
        self.outputType = argMinOptions.OutputType()

    # ArgMinOptionsT
    def Pack(self, builder):
        ArgMinOptionsStart(builder)
        ArgMinOptionsAddOutputType(builder, self.outputType)
        argMinOptions = ArgMinOptionsEnd(builder)
        return argMinOptions


class TransposeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsTransposeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TransposeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def TransposeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # TransposeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def TransposeOptionsStart(builder):
    builder.StartObject(0)


def TransposeOptionsEnd(builder):
    return builder.EndObject()


class TransposeOptionsT(object):

    # TransposeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        transposeOptions = TransposeOptions()
        transposeOptions.Init(buf, pos)
        return cls.InitFromObj(transposeOptions)

    @classmethod
    def InitFromObj(cls, transposeOptions):
        x = TransposeOptionsT()
        x._UnPack(transposeOptions)
        return x

    # TransposeOptionsT
    def _UnPack(self, transposeOptions):
        if transposeOptions is None:
            return

    # TransposeOptionsT
    def Pack(self, builder):
        TransposeOptionsStart(builder)
        transposeOptions = TransposeOptionsEnd(builder)
        return transposeOptions


class DimensionType(object):
    DENSE = 0
    SPARSE_CSR = 1


class Operator(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsOperator(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Operator()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def OperatorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Operator
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Operator
    def OpcodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Uint32Flags, o + self._tab.Pos
            )
        return 0

    # Operator
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Operator
    def InputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Operator
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Operator
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Operator
    def OutputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Operator
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Operator
    def BuiltinOptionsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Operator
    def BuiltinOptions(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            from flatbuffers.table import Table

            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # Operator
    def CustomOptions(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Uint8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # Operator
    def CustomOptionsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Operator
    def CustomOptionsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def CustomOptionsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Operator
    def CustomOptionsFormat(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Operator
    def MutatingVariableInputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.BoolFlags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # Operator
    def MutatingVariableInputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.BoolFlags, o)
        return 0

    # Operator
    def MutatingVariableInputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def MutatingVariableInputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Operator
    def Intermediates(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Operator
    def IntermediatesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Operator
    def IntermediatesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def IntermediatesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0


def OperatorStart(builder):
    builder.StartObject(9)


def OperatorAddOpcodeIndex(builder, opcodeIndex):
    builder.PrependUint32Slot(0, opcodeIndex, 0)


def OperatorAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0
    )


def OperatorStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def OperatorAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0
    )


def OperatorStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def OperatorAddBuiltinOptionsType(builder, builtinOptionsType):
    builder.PrependUint8Slot(3, builtinOptionsType, 0)


def OperatorAddBuiltinOptions(builder, builtinOptions):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(builtinOptions), 0
    )


def OperatorAddCustomOptions(builder, customOptions):
    builder.PrependUOffsetTRelativeSlot(
        5, flatbuffers.number_types.UOffsetTFlags.py_type(customOptions), 0
    )


def OperatorStartCustomOptionsVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def OperatorAddCustomOptionsFormat(builder, customOptionsFormat):
    builder.PrependInt8Slot(6, customOptionsFormat, 0)


def OperatorAddMutatingVariableInputs(builder, mutatingVariableInputs):
    builder.PrependUOffsetTRelativeSlot(
        7, flatbuffers.number_types.UOffsetTFlags.py_type(mutatingVariableInputs), 0
    )


def OperatorStartMutatingVariableInputsVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def OperatorAddIntermediates(builder, intermediates):
    builder.PrependUOffsetTRelativeSlot(
        8, flatbuffers.number_types.UOffsetTFlags.py_type(intermediates), 0
    )


def OperatorStartIntermediatesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def OperatorEnd(builder):
    return builder.EndObject()


class OperatorT(object):

    # OperatorT
    def __init__(self):
        self.opcodeIndex = 0  # type: int
        self.inputs = None  # type: List[int]
        self.outputs = None  # type: List[int]
        self.builtinOptionsType = 0  # type: int
        self.builtinOptions = (
            None
        )  # type: Union[None, Conv2DOptionsT, DepthwiseConv2DOptionsT, ConcatEmbeddingsOptionsT, LSHProjectionOptionsT, Pool2DOptionsT, SVDFOptionsT, RNNOptionsT, FullyConnectedOptionsT, SoftmaxOptionsT, ConcatenationOptionsT, AddOptionsT, L2NormOptionsT, LocalResponseNormalizationOptionsT, LSTMOptionsT, ResizeBilinearOptionsT, CallOptionsT, ReshapeOptionsT, SkipGramOptionsT, SpaceToDepthOptionsT, EmbeddingLookupSparseOptionsT, MulOptionsT, PadOptionsT, GatherOptionsT, BatchToSpaceNDOptionsT, SpaceToBatchNDOptionsT, TransposeOptionsT, ReducerOptionsT, SubOptionsT, DivOptionsT, SqueezeOptionsT, SequenceRNNOptionsT, StridedSliceOptionsT, ExpOptionsT, TopKV2OptionsT, SplitOptionsT, LogSoftmaxOptionsT, CastOptionsT, DequantizeOptionsT, MaximumMinimumOptionsT, ArgMaxOptionsT, LessOptionsT, NegOptionsT, PadV2OptionsT, GreaterOptionsT, GreaterEqualOptionsT, LessEqualOptionsT, SelectOptionsT, SliceOptionsT, TransposeConvOptionsT, SparseToDenseOptionsT, TileOptionsT, ExpandDimsOptionsT, EqualOptionsT, NotEqualOptionsT, ShapeOptionsT, PowOptionsT, ArgMinOptionsT, FakeQuantOptionsT, PackOptionsT, LogicalOrOptionsT, OneHotOptionsT, LogicalAndOptionsT, LogicalNotOptionsT, UnpackOptionsT, FloorDivOptionsT, SquareOptionsT, ZerosLikeOptionsT, FillOptionsT, BidirectionalSequenceLSTMOptionsT, BidirectionalSequenceRNNOptionsT, UnidirectionalSequenceLSTMOptionsT, FloorModOptionsT, RangeOptionsT, ResizeNearestNeighborOptionsT, LeakyReluOptionsT, SquaredDifferenceOptionsT, MirrorPadOptionsT, AbsOptionsT, SplitVOptionsT, UniqueOptionsT, ReverseV2OptionsT, AddNOptionsT, GatherNdOptionsT, CosOptionsT, WhereOptionsT, RankOptionsT, ReverseSequenceOptionsT, MatrixDiagOptionsT, QuantizeOptionsT, MatrixSetDiagOptionsT, HardSwishOptionsT, IfOptionsT, WhileOptionsT, DepthToSpaceOptionsT, NonMaxSuppressionV4OptionsT, NonMaxSuppressionV5OptionsT, ScatterNdOptionsT, SelectV2OptionsT, DensifyOptionsT, SegmentSumOptionsT, BatchMatMulOptionsT]
        self.customOptions = None  # type: List[int]
        self.customOptionsFormat = 0  # type: int
        self.mutatingVariableInputs = None  # type: List[bool]
        self.intermediates = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operator = Operator()
        operator.Init(buf, pos)
        return cls.InitFromObj(operator)

    @classmethod
    def InitFromObj(cls, operator):
        x = OperatorT()
        x._UnPack(operator)
        return x

    # OperatorT
    def _UnPack(self, operator):
        if operator is None:
            return
        self.opcodeIndex = operator.OpcodeIndex()
        if not operator.InputsIsNone():
            if np is None:
                self.inputs = []
                for i in range(operator.InputsLength()):
                    self.inputs.append(operator.Inputs(i))
            else:
                self.inputs = operator.InputsAsNumpy()
        if not operator.OutputsIsNone():
            if np is None:
                self.outputs = []
                for i in range(operator.OutputsLength()):
                    self.outputs.append(operator.Outputs(i))
            else:
                self.outputs = operator.OutputsAsNumpy()
        self.builtinOptionsType = operator.BuiltinOptionsType()
        self.builtinOptions = BuiltinOptionsCreator(
            self.builtinOptionsType, operator.BuiltinOptions()
        )
        if not operator.CustomOptionsIsNone():
            if np is None:
                self.customOptions = []
                for i in range(operator.CustomOptionsLength()):
                    self.customOptions.append(operator.CustomOptions(i))
            else:
                self.customOptions = operator.CustomOptionsAsNumpy()
        self.customOptionsFormat = operator.CustomOptionsFormat()
        if not operator.MutatingVariableInputsIsNone():
            if np is None:
                self.mutatingVariableInputs = []
                for i in range(operator.MutatingVariableInputsLength()):
                    self.mutatingVariableInputs.append(
                        operator.MutatingVariableInputs(i)
                    )
            else:
                self.mutatingVariableInputs = operator.MutatingVariableInputsAsNumpy()
        if not operator.IntermediatesIsNone():
            if np is None:
                self.intermediates = []
                for i in range(operator.IntermediatesLength()):
                    self.intermediates.append(operator.Intermediates(i))
            else:
                self.intermediates = operator.IntermediatesAsNumpy()

    # OperatorT
    def Pack(self, builder):
        if self.inputs is not None:
            if np is not None and type(self.inputs) is np.ndarray:
                inputs = builder.CreateNumpyVector(self.inputs)
            else:
                OperatorStartInputsVector(builder, len(self.inputs))
                for i in reversed(range(len(self.inputs))):
                    builder.PrependInt32(self.inputs[i])
                inputs = builder.EndVector(len(self.inputs))
        if self.outputs is not None:
            if np is not None and type(self.outputs) is np.ndarray:
                outputs = builder.CreateNumpyVector(self.outputs)
            else:
                OperatorStartOutputsVector(builder, len(self.outputs))
                for i in reversed(range(len(self.outputs))):
                    builder.PrependInt32(self.outputs[i])
                outputs = builder.EndVector(len(self.outputs))
        if self.builtinOptions is not None:
            builtinOptions = self.builtinOptions.Pack(builder)
        if self.customOptions is not None:
            if np is not None and type(self.customOptions) is np.ndarray:
                customOptions = builder.CreateNumpyVector(self.customOptions)
            else:
                OperatorStartCustomOptionsVector(builder, len(self.customOptions))
                for i in reversed(range(len(self.customOptions))):
                    builder.PrependUint8(self.customOptions[i])
                customOptions = builder.EndVector(len(self.customOptions))
        if self.mutatingVariableInputs is not None:
            if np is not None and type(self.mutatingVariableInputs) is np.ndarray:
                mutatingVariableInputs = builder.CreateNumpyVector(
                    self.mutatingVariableInputs
                )
            else:
                OperatorStartMutatingVariableInputsVector(
                    builder, len(self.mutatingVariableInputs)
                )
                for i in reversed(range(len(self.mutatingVariableInputs))):
                    builder.PrependBool(self.mutatingVariableInputs[i])
                mutatingVariableInputs = builder.EndVector(
                    len(self.mutatingVariableInputs)
                )
        if self.intermediates is not None:
            if np is not None and type(self.intermediates) is np.ndarray:
                intermediates = builder.CreateNumpyVector(self.intermediates)
            else:
                OperatorStartIntermediatesVector(builder, len(self.intermediates))
                for i in reversed(range(len(self.intermediates))):
                    builder.PrependInt32(self.intermediates[i])
                intermediates = builder.EndVector(len(self.intermediates))
        OperatorStart(builder)
        OperatorAddOpcodeIndex(builder, self.opcodeIndex)
        if self.inputs is not None:
            OperatorAddInputs(builder, inputs)
        if self.outputs is not None:
            OperatorAddOutputs(builder, outputs)
        OperatorAddBuiltinOptionsType(builder, self.builtinOptionsType)
        if self.builtinOptions is not None:
            OperatorAddBuiltinOptions(builder, builtinOptions)
        if self.customOptions is not None:
            OperatorAddCustomOptions(builder, customOptions)
        OperatorAddCustomOptionsFormat(builder, self.customOptionsFormat)
        if self.mutatingVariableInputs is not None:
            OperatorAddMutatingVariableInputs(builder, mutatingVariableInputs)
        if self.intermediates is not None:
            OperatorAddIntermediates(builder, intermediates)
        operator = OperatorEnd(builder)
        return operator


class FakeQuantOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsFakeQuantOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FakeQuantOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FakeQuantOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # FakeQuantOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FakeQuantOptions
    def Min(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # FakeQuantOptions
    def Max(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # FakeQuantOptions
    def NumBits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FakeQuantOptions
    def NarrowRange(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def FakeQuantOptionsStart(builder):
    builder.StartObject(4)


def FakeQuantOptionsAddMin(builder, min):
    builder.PrependFloat32Slot(0, min, 0.0)


def FakeQuantOptionsAddMax(builder, max):
    builder.PrependFloat32Slot(1, max, 0.0)


def FakeQuantOptionsAddNumBits(builder, numBits):
    builder.PrependInt32Slot(2, numBits, 0)


def FakeQuantOptionsAddNarrowRange(builder, narrowRange):
    builder.PrependBoolSlot(3, narrowRange, 0)


def FakeQuantOptionsEnd(builder):
    return builder.EndObject()


class FakeQuantOptionsT(object):

    # FakeQuantOptionsT
    def __init__(self):
        self.min = 0.0  # type: float
        self.max = 0.0  # type: float
        self.numBits = 0  # type: int
        self.narrowRange = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fakeQuantOptions = FakeQuantOptions()
        fakeQuantOptions.Init(buf, pos)
        return cls.InitFromObj(fakeQuantOptions)

    @classmethod
    def InitFromObj(cls, fakeQuantOptions):
        x = FakeQuantOptionsT()
        x._UnPack(fakeQuantOptions)
        return x

    # FakeQuantOptionsT
    def _UnPack(self, fakeQuantOptions):
        if fakeQuantOptions is None:
            return
        self.min = fakeQuantOptions.Min()
        self.max = fakeQuantOptions.Max()
        self.numBits = fakeQuantOptions.NumBits()
        self.narrowRange = fakeQuantOptions.NarrowRange()

    # FakeQuantOptionsT
    def Pack(self, builder):
        FakeQuantOptionsStart(builder)
        FakeQuantOptionsAddMin(builder, self.min)
        FakeQuantOptionsAddMax(builder, self.max)
        FakeQuantOptionsAddNumBits(builder, self.numBits)
        FakeQuantOptionsAddNarrowRange(builder, self.narrowRange)
        fakeQuantOptions = FakeQuantOptionsEnd(builder)
        return fakeQuantOptions


class MaximumMinimumOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMaximumMinimumOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MaximumMinimumOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MaximumMinimumOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # MaximumMinimumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def MaximumMinimumOptionsStart(builder):
    builder.StartObject(0)


def MaximumMinimumOptionsEnd(builder):
    return builder.EndObject()


class MaximumMinimumOptionsT(object):

    # MaximumMinimumOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        maximumMinimumOptions = MaximumMinimumOptions()
        maximumMinimumOptions.Init(buf, pos)
        return cls.InitFromObj(maximumMinimumOptions)

    @classmethod
    def InitFromObj(cls, maximumMinimumOptions):
        x = MaximumMinimumOptionsT()
        x._UnPack(maximumMinimumOptions)
        return x

    # MaximumMinimumOptionsT
    def _UnPack(self, maximumMinimumOptions):
        if maximumMinimumOptions is None:
            return

    # MaximumMinimumOptionsT
    def Pack(self, builder):
        MaximumMinimumOptionsStart(builder)
        maximumMinimumOptions = MaximumMinimumOptionsEnd(builder)
        return maximumMinimumOptions


class TensorType(object):
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9
    FLOAT64 = 10
    COMPLEX128 = 11


class SegmentSumOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSegmentSumOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SegmentSumOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SegmentSumOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SegmentSumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SegmentSumOptionsStart(builder):
    builder.StartObject(0)


def SegmentSumOptionsEnd(builder):
    return builder.EndObject()


class SegmentSumOptionsT(object):

    # SegmentSumOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        segmentSumOptions = SegmentSumOptions()
        segmentSumOptions.Init(buf, pos)
        return cls.InitFromObj(segmentSumOptions)

    @classmethod
    def InitFromObj(cls, segmentSumOptions):
        x = SegmentSumOptionsT()
        x._UnPack(segmentSumOptions)
        return x

    # SegmentSumOptionsT
    def _UnPack(self, segmentSumOptions):
        if segmentSumOptions is None:
            return

    # SegmentSumOptionsT
    def Pack(self, builder):
        SegmentSumOptionsStart(builder)
        segmentSumOptions = SegmentSumOptionsEnd(builder)
        return segmentSumOptions


class EmbeddingLookupSparseOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsEmbeddingLookupSparseOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EmbeddingLookupSparseOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def EmbeddingLookupSparseOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # EmbeddingLookupSparseOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EmbeddingLookupSparseOptions
    def Combiner(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def EmbeddingLookupSparseOptionsStart(builder):
    builder.StartObject(1)


def EmbeddingLookupSparseOptionsAddCombiner(builder, combiner):
    builder.PrependInt8Slot(0, combiner, 0)


def EmbeddingLookupSparseOptionsEnd(builder):
    return builder.EndObject()


class EmbeddingLookupSparseOptionsT(object):

    # EmbeddingLookupSparseOptionsT
    def __init__(self):
        self.combiner = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        embeddingLookupSparseOptions = EmbeddingLookupSparseOptions()
        embeddingLookupSparseOptions.Init(buf, pos)
        return cls.InitFromObj(embeddingLookupSparseOptions)

    @classmethod
    def InitFromObj(cls, embeddingLookupSparseOptions):
        x = EmbeddingLookupSparseOptionsT()
        x._UnPack(embeddingLookupSparseOptions)
        return x

    # EmbeddingLookupSparseOptionsT
    def _UnPack(self, embeddingLookupSparseOptions):
        if embeddingLookupSparseOptions is None:
            return
        self.combiner = embeddingLookupSparseOptions.Combiner()

    # EmbeddingLookupSparseOptionsT
    def Pack(self, builder):
        EmbeddingLookupSparseOptionsStart(builder)
        EmbeddingLookupSparseOptionsAddCombiner(builder, self.combiner)
        embeddingLookupSparseOptions = EmbeddingLookupSparseOptionsEnd(builder)
        return embeddingLookupSparseOptions


class FullyConnectedOptionsWeightsFormat(object):
    DEFAULT = 0
    SHUFFLED4x16INT8 = 1


class UnidirectionalSequenceLSTMOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsUnidirectionalSequenceLSTMOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnidirectionalSequenceLSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def UnidirectionalSequenceLSTMOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # UnidirectionalSequenceLSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UnidirectionalSequenceLSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # UnidirectionalSequenceLSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # UnidirectionalSequenceLSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # UnidirectionalSequenceLSTMOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # UnidirectionalSequenceLSTMOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def UnidirectionalSequenceLSTMOptionsStart(builder):
    builder.StartObject(5)


def UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
    builder, fusedActivationFunction
):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def UnidirectionalSequenceLSTMOptionsAddCellClip(builder, cellClip):
    builder.PrependFloat32Slot(1, cellClip, 0.0)


def UnidirectionalSequenceLSTMOptionsAddProjClip(builder, projClip):
    builder.PrependFloat32Slot(2, projClip, 0.0)


def UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(3, timeMajor, 0)


def UnidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(
    builder, asymmetricQuantizeInputs
):
    builder.PrependBoolSlot(4, asymmetricQuantizeInputs, 0)


def UnidirectionalSequenceLSTMOptionsEnd(builder):
    return builder.EndObject()


class UnidirectionalSequenceLSTMOptionsT(object):

    # UnidirectionalSequenceLSTMOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.cellClip = 0.0  # type: float
        self.projClip = 0.0  # type: float
        self.timeMajor = False  # type: bool
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unidirectionalSequenceLSTMOptions = UnidirectionalSequenceLSTMOptions()
        unidirectionalSequenceLSTMOptions.Init(buf, pos)
        return cls.InitFromObj(unidirectionalSequenceLSTMOptions)

    @classmethod
    def InitFromObj(cls, unidirectionalSequenceLSTMOptions):
        x = UnidirectionalSequenceLSTMOptionsT()
        x._UnPack(unidirectionalSequenceLSTMOptions)
        return x

    # UnidirectionalSequenceLSTMOptionsT
    def _UnPack(self, unidirectionalSequenceLSTMOptions):
        if unidirectionalSequenceLSTMOptions is None:
            return
        self.fusedActivationFunction = (
            unidirectionalSequenceLSTMOptions.FusedActivationFunction()
        )
        self.cellClip = unidirectionalSequenceLSTMOptions.CellClip()
        self.projClip = unidirectionalSequenceLSTMOptions.ProjClip()
        self.timeMajor = unidirectionalSequenceLSTMOptions.TimeMajor()
        self.asymmetricQuantizeInputs = (
            unidirectionalSequenceLSTMOptions.AsymmetricQuantizeInputs()
        )

    # UnidirectionalSequenceLSTMOptionsT
    def Pack(self, builder):
        UnidirectionalSequenceLSTMOptionsStart(builder)
        UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        UnidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        UnidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        UnidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(
            builder, self.asymmetricQuantizeInputs
        )
        unidirectionalSequenceLSTMOptions = UnidirectionalSequenceLSTMOptionsEnd(
            builder
        )
        return unidirectionalSequenceLSTMOptions


class ResizeNearestNeighborOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsResizeNearestNeighborOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeNearestNeighborOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ResizeNearestNeighborOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ResizeNearestNeighborOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ResizeNearestNeighborOptions
    def AlignCorners(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # ResizeNearestNeighborOptions
    def HalfPixelCenters(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def ResizeNearestNeighborOptionsStart(builder):
    builder.StartObject(2)


def ResizeNearestNeighborOptionsAddAlignCorners(builder, alignCorners):
    builder.PrependBoolSlot(0, alignCorners, 0)


def ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, halfPixelCenters):
    builder.PrependBoolSlot(1, halfPixelCenters, 0)


def ResizeNearestNeighborOptionsEnd(builder):
    return builder.EndObject()


class ResizeNearestNeighborOptionsT(object):

    # ResizeNearestNeighborOptionsT
    def __init__(self):
        self.alignCorners = False  # type: bool
        self.halfPixelCenters = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        resizeNearestNeighborOptions = ResizeNearestNeighborOptions()
        resizeNearestNeighborOptions.Init(buf, pos)
        return cls.InitFromObj(resizeNearestNeighborOptions)

    @classmethod
    def InitFromObj(cls, resizeNearestNeighborOptions):
        x = ResizeNearestNeighborOptionsT()
        x._UnPack(resizeNearestNeighborOptions)
        return x

    # ResizeNearestNeighborOptionsT
    def _UnPack(self, resizeNearestNeighborOptions):
        if resizeNearestNeighborOptions is None:
            return
        self.alignCorners = resizeNearestNeighborOptions.AlignCorners()
        self.halfPixelCenters = resizeNearestNeighborOptions.HalfPixelCenters()

    # ResizeNearestNeighborOptionsT
    def Pack(self, builder):
        ResizeNearestNeighborOptionsStart(builder)
        ResizeNearestNeighborOptionsAddAlignCorners(builder, self.alignCorners)
        ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        resizeNearestNeighborOptions = ResizeNearestNeighborOptionsEnd(builder)
        return resizeNearestNeighborOptions


class CastOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsCastOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CastOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CastOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # CastOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CastOptions
    def InDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # CastOptions
    def OutDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def CastOptionsStart(builder):
    builder.StartObject(2)


def CastOptionsAddInDataType(builder, inDataType):
    builder.PrependInt8Slot(0, inDataType, 0)


def CastOptionsAddOutDataType(builder, outDataType):
    builder.PrependInt8Slot(1, outDataType, 0)


def CastOptionsEnd(builder):
    return builder.EndObject()


class CastOptionsT(object):

    # CastOptionsT
    def __init__(self):
        self.inDataType = 0  # type: int
        self.outDataType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        castOptions = CastOptions()
        castOptions.Init(buf, pos)
        return cls.InitFromObj(castOptions)

    @classmethod
    def InitFromObj(cls, castOptions):
        x = CastOptionsT()
        x._UnPack(castOptions)
        return x

    # CastOptionsT
    def _UnPack(self, castOptions):
        if castOptions is None:
            return
        self.inDataType = castOptions.InDataType()
        self.outDataType = castOptions.OutDataType()

    # CastOptionsT
    def Pack(self, builder):
        CastOptionsStart(builder)
        CastOptionsAddInDataType(builder, self.inDataType)
        CastOptionsAddOutDataType(builder, self.outDataType)
        castOptions = CastOptionsEnd(builder)
        return castOptions


class BidirectionalSequenceLSTMOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsBidirectionalSequenceLSTMOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BidirectionalSequenceLSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BidirectionalSequenceLSTMOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # BidirectionalSequenceLSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BidirectionalSequenceLSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # BidirectionalSequenceLSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # BidirectionalSequenceLSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # BidirectionalSequenceLSTMOptions
    def MergeOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # BidirectionalSequenceLSTMOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return True

    # BidirectionalSequenceLSTMOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def BidirectionalSequenceLSTMOptionsStart(builder):
    builder.StartObject(6)


def BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
    builder, fusedActivationFunction
):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def BidirectionalSequenceLSTMOptionsAddCellClip(builder, cellClip):
    builder.PrependFloat32Slot(1, cellClip, 0.0)


def BidirectionalSequenceLSTMOptionsAddProjClip(builder, projClip):
    builder.PrependFloat32Slot(2, projClip, 0.0)


def BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, mergeOutputs):
    builder.PrependBoolSlot(3, mergeOutputs, 0)


def BidirectionalSequenceLSTMOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(4, timeMajor, 1)


def BidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(
    builder, asymmetricQuantizeInputs
):
    builder.PrependBoolSlot(5, asymmetricQuantizeInputs, 0)


def BidirectionalSequenceLSTMOptionsEnd(builder):
    return builder.EndObject()


class BidirectionalSequenceLSTMOptionsT(object):

    # BidirectionalSequenceLSTMOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.cellClip = 0.0  # type: float
        self.projClip = 0.0  # type: float
        self.mergeOutputs = False  # type: bool
        self.timeMajor = True  # type: bool
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bidirectionalSequenceLSTMOptions = BidirectionalSequenceLSTMOptions()
        bidirectionalSequenceLSTMOptions.Init(buf, pos)
        return cls.InitFromObj(bidirectionalSequenceLSTMOptions)

    @classmethod
    def InitFromObj(cls, bidirectionalSequenceLSTMOptions):
        x = BidirectionalSequenceLSTMOptionsT()
        x._UnPack(bidirectionalSequenceLSTMOptions)
        return x

    # BidirectionalSequenceLSTMOptionsT
    def _UnPack(self, bidirectionalSequenceLSTMOptions):
        if bidirectionalSequenceLSTMOptions is None:
            return
        self.fusedActivationFunction = (
            bidirectionalSequenceLSTMOptions.FusedActivationFunction()
        )
        self.cellClip = bidirectionalSequenceLSTMOptions.CellClip()
        self.projClip = bidirectionalSequenceLSTMOptions.ProjClip()
        self.mergeOutputs = bidirectionalSequenceLSTMOptions.MergeOutputs()
        self.timeMajor = bidirectionalSequenceLSTMOptions.TimeMajor()
        self.asymmetricQuantizeInputs = (
            bidirectionalSequenceLSTMOptions.AsymmetricQuantizeInputs()
        )

    # BidirectionalSequenceLSTMOptionsT
    def Pack(self, builder):
        BidirectionalSequenceLSTMOptionsStart(builder)
        BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        BidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        BidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, self.mergeOutputs)
        BidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        BidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(
            builder, self.asymmetricQuantizeInputs
        )
        bidirectionalSequenceLSTMOptions = BidirectionalSequenceLSTMOptionsEnd(builder)
        return bidirectionalSequenceLSTMOptions


class SplitOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSplitOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SplitOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SplitOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SplitOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SplitOptions
    def NumSplits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def SplitOptionsStart(builder):
    builder.StartObject(1)


def SplitOptionsAddNumSplits(builder, numSplits):
    builder.PrependInt32Slot(0, numSplits, 0)


def SplitOptionsEnd(builder):
    return builder.EndObject()


class SplitOptionsT(object):

    # SplitOptionsT
    def __init__(self):
        self.numSplits = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        splitOptions = SplitOptions()
        splitOptions.Init(buf, pos)
        return cls.InitFromObj(splitOptions)

    @classmethod
    def InitFromObj(cls, splitOptions):
        x = SplitOptionsT()
        x._UnPack(splitOptions)
        return x

    # SplitOptionsT
    def _UnPack(self, splitOptions):
        if splitOptions is None:
            return
        self.numSplits = splitOptions.NumSplits()

    # SplitOptionsT
    def Pack(self, builder):
        SplitOptionsStart(builder)
        SplitOptionsAddNumSplits(builder, self.numSplits)
        splitOptions = SplitOptionsEnd(builder)
        return splitOptions


class BuiltinOptions(object):
    NONE = 0
    Conv2DOptions = 1
    DepthwiseConv2DOptions = 2
    ConcatEmbeddingsOptions = 3
    LSHProjectionOptions = 4
    Pool2DOptions = 5
    SVDFOptions = 6
    RNNOptions = 7
    FullyConnectedOptions = 8
    SoftmaxOptions = 9
    ConcatenationOptions = 10
    AddOptions = 11
    L2NormOptions = 12
    LocalResponseNormalizationOptions = 13
    LSTMOptions = 14
    ResizeBilinearOptions = 15
    CallOptions = 16
    ReshapeOptions = 17
    SkipGramOptions = 18
    SpaceToDepthOptions = 19
    EmbeddingLookupSparseOptions = 20
    MulOptions = 21
    PadOptions = 22
    GatherOptions = 23
    BatchToSpaceNDOptions = 24
    SpaceToBatchNDOptions = 25
    TransposeOptions = 26
    ReducerOptions = 27
    SubOptions = 28
    DivOptions = 29
    SqueezeOptions = 30
    SequenceRNNOptions = 31
    StridedSliceOptions = 32
    ExpOptions = 33
    TopKV2Options = 34
    SplitOptions = 35
    LogSoftmaxOptions = 36
    CastOptions = 37
    DequantizeOptions = 38
    MaximumMinimumOptions = 39
    ArgMaxOptions = 40
    LessOptions = 41
    NegOptions = 42
    PadV2Options = 43
    GreaterOptions = 44
    GreaterEqualOptions = 45
    LessEqualOptions = 46
    SelectOptions = 47
    SliceOptions = 48
    TransposeConvOptions = 49
    SparseToDenseOptions = 50
    TileOptions = 51
    ExpandDimsOptions = 52
    EqualOptions = 53
    NotEqualOptions = 54
    ShapeOptions = 55
    PowOptions = 56
    ArgMinOptions = 57
    FakeQuantOptions = 58
    PackOptions = 59
    LogicalOrOptions = 60
    OneHotOptions = 61
    LogicalAndOptions = 62
    LogicalNotOptions = 63
    UnpackOptions = 64
    FloorDivOptions = 65
    SquareOptions = 66
    ZerosLikeOptions = 67
    FillOptions = 68
    BidirectionalSequenceLSTMOptions = 69
    BidirectionalSequenceRNNOptions = 70
    UnidirectionalSequenceLSTMOptions = 71
    FloorModOptions = 72
    RangeOptions = 73
    ResizeNearestNeighborOptions = 74
    LeakyReluOptions = 75
    SquaredDifferenceOptions = 76
    MirrorPadOptions = 77
    AbsOptions = 78
    SplitVOptions = 79
    UniqueOptions = 80
    ReverseV2Options = 81
    AddNOptions = 82
    GatherNdOptions = 83
    CosOptions = 84
    WhereOptions = 85
    RankOptions = 86
    ReverseSequenceOptions = 87
    MatrixDiagOptions = 88
    QuantizeOptions = 89
    MatrixSetDiagOptions = 90
    HardSwishOptions = 91
    IfOptions = 92
    WhileOptions = 93
    DepthToSpaceOptions = 94
    NonMaxSuppressionV4Options = 95
    NonMaxSuppressionV5Options = 96
    ScatterNdOptions = 97
    SelectV2Options = 98
    DensifyOptions = 99
    SegmentSumOptions = 100
    BatchMatMulOptions = 101


def BuiltinOptionsCreator(unionType, table):
    from flatbuffers.table import Table

    if not isinstance(table, Table):
        return None
    if unionType == BuiltinOptions().Conv2DOptions:
        return Conv2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DepthwiseConv2DOptions:
        return DepthwiseConv2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ConcatEmbeddingsOptions:
        return ConcatEmbeddingsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LSHProjectionOptions:
        return LSHProjectionOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().Pool2DOptions:
        return Pool2DOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SVDFOptions:
        return SVDFOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RNNOptions:
        return RNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FullyConnectedOptions:
        return FullyConnectedOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SoftmaxOptions:
        return SoftmaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ConcatenationOptions:
        return ConcatenationOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AddOptions:
        return AddOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().L2NormOptions:
        return L2NormOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LocalResponseNormalizationOptions:
        return LocalResponseNormalizationOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LSTMOptions:
        return LSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ResizeBilinearOptions:
        return ResizeBilinearOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CallOptions:
        return CallOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReshapeOptions:
        return ReshapeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SkipGramOptions:
        return SkipGramOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SpaceToDepthOptions:
        return SpaceToDepthOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().EmbeddingLookupSparseOptions:
        return EmbeddingLookupSparseOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MulOptions:
        return MulOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PadOptions:
        return PadOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GatherOptions:
        return GatherOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BatchToSpaceNDOptions:
        return BatchToSpaceNDOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SpaceToBatchNDOptions:
        return SpaceToBatchNDOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TransposeOptions:
        return TransposeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReducerOptions:
        return ReducerOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SubOptions:
        return SubOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DivOptions:
        return DivOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SqueezeOptions:
        return SqueezeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SequenceRNNOptions:
        return SequenceRNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().StridedSliceOptions:
        return StridedSliceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ExpOptions:
        return ExpOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TopKV2Options:
        return TopKV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SplitOptions:
        return SplitOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogSoftmaxOptions:
        return LogSoftmaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CastOptions:
        return CastOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DequantizeOptions:
        return DequantizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MaximumMinimumOptions:
        return MaximumMinimumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ArgMaxOptions:
        return ArgMaxOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LessOptions:
        return LessOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NegOptions:
        return NegOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PadV2Options:
        return PadV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GreaterOptions:
        return GreaterOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GreaterEqualOptions:
        return GreaterEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LessEqualOptions:
        return LessEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SelectOptions:
        return SelectOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SliceOptions:
        return SliceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TransposeConvOptions:
        return TransposeConvOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SparseToDenseOptions:
        return SparseToDenseOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().TileOptions:
        return TileOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ExpandDimsOptions:
        return ExpandDimsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().EqualOptions:
        return EqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NotEqualOptions:
        return NotEqualOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ShapeOptions:
        return ShapeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PowOptions:
        return PowOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ArgMinOptions:
        return ArgMinOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FakeQuantOptions:
        return FakeQuantOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().PackOptions:
        return PackOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalOrOptions:
        return LogicalOrOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().OneHotOptions:
        return OneHotOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalAndOptions:
        return LogicalAndOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LogicalNotOptions:
        return LogicalNotOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnpackOptions:
        return UnpackOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FloorDivOptions:
        return FloorDivOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SquareOptions:
        return SquareOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ZerosLikeOptions:
        return ZerosLikeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FillOptions:
        return FillOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BidirectionalSequenceLSTMOptions:
        return BidirectionalSequenceLSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BidirectionalSequenceRNNOptions:
        return BidirectionalSequenceRNNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UnidirectionalSequenceLSTMOptions:
        return UnidirectionalSequenceLSTMOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().FloorModOptions:
        return FloorModOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RangeOptions:
        return RangeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ResizeNearestNeighborOptions:
        return ResizeNearestNeighborOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().LeakyReluOptions:
        return LeakyReluOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SquaredDifferenceOptions:
        return SquaredDifferenceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MirrorPadOptions:
        return MirrorPadOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AbsOptions:
        return AbsOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SplitVOptions:
        return SplitVOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().UniqueOptions:
        return UniqueOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReverseV2Options:
        return ReverseV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().AddNOptions:
        return AddNOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().GatherNdOptions:
        return GatherNdOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().CosOptions:
        return CosOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().WhereOptions:
        return WhereOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().RankOptions:
        return RankOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ReverseSequenceOptions:
        return ReverseSequenceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MatrixDiagOptions:
        return MatrixDiagOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().QuantizeOptions:
        return QuantizeOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().MatrixSetDiagOptions:
        return MatrixSetDiagOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().HardSwishOptions:
        return HardSwishOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().IfOptions:
        return IfOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().WhileOptions:
        return WhileOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DepthToSpaceOptions:
        return DepthToSpaceOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NonMaxSuppressionV4Options:
        return NonMaxSuppressionV4OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().NonMaxSuppressionV5Options:
        return NonMaxSuppressionV5OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().ScatterNdOptions:
        return ScatterNdOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SelectV2Options:
        return SelectV2OptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().DensifyOptions:
        return DensifyOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().SegmentSumOptions:
        return SegmentSumOptionsT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == BuiltinOptions().BatchMatMulOptions:
        return BatchMatMulOptionsT.InitFromBuf(table.Bytes, table.Pos)
    return None


class QuantizationParameters(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsQuantizationParameters(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = QuantizationParameters()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def QuantizationParametersBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # QuantizationParameters
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # QuantizationParameters
    def Min(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # QuantizationParameters
    def MinAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def MinLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def MinIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # QuantizationParameters
    def Max(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # QuantizationParameters
    def MaxAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def MaxLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def MaxIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # QuantizationParameters
    def Scale(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # QuantizationParameters
    def ScaleAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # QuantizationParameters
    def ScaleLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def ScaleIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # QuantizationParameters
    def ZeroPoint(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int64Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8),
            )
        return 0

    # QuantizationParameters
    def ZeroPointAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # QuantizationParameters
    def ZeroPointLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # QuantizationParameters
    def ZeroPointIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # QuantizationParameters
    def DetailsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # QuantizationParameters
    def Details(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            from flatbuffers.table import Table

            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # QuantizationParameters
    def QuantizedDimension(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def QuantizationParametersStart(builder):
    builder.StartObject(7)


def QuantizationParametersAddMin(builder, min):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(min), 0
    )


def QuantizationParametersStartMinVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def QuantizationParametersAddMax(builder, max):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(max), 0
    )


def QuantizationParametersStartMaxVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def QuantizationParametersAddScale(builder, scale):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(scale), 0
    )


def QuantizationParametersStartScaleVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def QuantizationParametersAddZeroPoint(builder, zeroPoint):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(zeroPoint), 0
    )


def QuantizationParametersStartZeroPointVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)


def QuantizationParametersAddDetailsType(builder, detailsType):
    builder.PrependUint8Slot(4, detailsType, 0)


def QuantizationParametersAddDetails(builder, details):
    builder.PrependUOffsetTRelativeSlot(
        5, flatbuffers.number_types.UOffsetTFlags.py_type(details), 0
    )


def QuantizationParametersAddQuantizedDimension(builder, quantizedDimension):
    builder.PrependInt32Slot(6, quantizedDimension, 0)


def QuantizationParametersEnd(builder):
    return builder.EndObject()


class QuantizationParametersT(object):

    # QuantizationParametersT
    def __init__(self):
        self.min = None  # type: List[float]
        self.max = None  # type: List[float]
        self.scale = None  # type: List[float]
        self.zeroPoint = None  # type: List[int]
        self.detailsType = 0  # type: int
        self.details = None  # type: Union[None, CustomQuantizationT]
        self.quantizedDimension = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quantizationParameters = QuantizationParameters()
        quantizationParameters.Init(buf, pos)
        return cls.InitFromObj(quantizationParameters)

    @classmethod
    def InitFromObj(cls, quantizationParameters):
        x = QuantizationParametersT()
        x._UnPack(quantizationParameters)
        return x

    # QuantizationParametersT
    def _UnPack(self, quantizationParameters):
        if quantizationParameters is None:
            return
        if not quantizationParameters.MinIsNone():
            if np is None:
                self.min = []
                for i in range(quantizationParameters.MinLength()):
                    self.min.append(quantizationParameters.Min(i))
            else:
                self.min = quantizationParameters.MinAsNumpy()
        if not quantizationParameters.MaxIsNone():
            if np is None:
                self.max = []
                for i in range(quantizationParameters.MaxLength()):
                    self.max.append(quantizationParameters.Max(i))
            else:
                self.max = quantizationParameters.MaxAsNumpy()
        if not quantizationParameters.ScaleIsNone():
            if np is None:
                self.scale = []
                for i in range(quantizationParameters.ScaleLength()):
                    self.scale.append(quantizationParameters.Scale(i))
            else:
                self.scale = quantizationParameters.ScaleAsNumpy()
        if not quantizationParameters.ZeroPointIsNone():
            if np is None:
                self.zeroPoint = []
                for i in range(quantizationParameters.ZeroPointLength()):
                    self.zeroPoint.append(quantizationParameters.ZeroPoint(i))
            else:
                self.zeroPoint = quantizationParameters.ZeroPointAsNumpy()
        self.detailsType = quantizationParameters.DetailsType()
        self.details = QuantizationDetailsCreator(
            self.detailsType, quantizationParameters.Details()
        )
        self.quantizedDimension = quantizationParameters.QuantizedDimension()

    # QuantizationParametersT
    def Pack(self, builder):
        if self.min is not None:
            if np is not None and type(self.min) is np.ndarray:
                min = builder.CreateNumpyVector(self.min)
            else:
                QuantizationParametersStartMinVector(builder, len(self.min))
                for i in reversed(range(len(self.min))):
                    builder.PrependFloat32(self.min[i])
                min = builder.EndVector(len(self.min))
        if self.max is not None:
            if np is not None and type(self.max) is np.ndarray:
                max = builder.CreateNumpyVector(self.max)
            else:
                QuantizationParametersStartMaxVector(builder, len(self.max))
                for i in reversed(range(len(self.max))):
                    builder.PrependFloat32(self.max[i])
                max = builder.EndVector(len(self.max))
        if self.scale is not None:
            if np is not None and type(self.scale) is np.ndarray:
                scale = builder.CreateNumpyVector(self.scale)
            else:
                QuantizationParametersStartScaleVector(builder, len(self.scale))
                for i in reversed(range(len(self.scale))):
                    builder.PrependFloat32(self.scale[i])
                scale = builder.EndVector(len(self.scale))
        if self.zeroPoint is not None:
            if np is not None and type(self.zeroPoint) is np.ndarray:
                zeroPoint = builder.CreateNumpyVector(self.zeroPoint)
            else:
                QuantizationParametersStartZeroPointVector(builder, len(self.zeroPoint))
                for i in reversed(range(len(self.zeroPoint))):
                    builder.PrependInt64(self.zeroPoint[i])
                zeroPoint = builder.EndVector(len(self.zeroPoint))
        if self.details is not None:
            details = self.details.Pack(builder)
        QuantizationParametersStart(builder)
        if self.min is not None:
            QuantizationParametersAddMin(builder, min)
        if self.max is not None:
            QuantizationParametersAddMax(builder, max)
        if self.scale is not None:
            QuantizationParametersAddScale(builder, scale)
        if self.zeroPoint is not None:
            QuantizationParametersAddZeroPoint(builder, zeroPoint)
        QuantizationParametersAddDetailsType(builder, self.detailsType)
        if self.details is not None:
            QuantizationParametersAddDetails(builder, details)
        QuantizationParametersAddQuantizedDimension(builder, self.quantizedDimension)
        quantizationParameters = QuantizationParametersEnd(builder)
        return quantizationParameters


class BuiltinOperator(object):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115
    ROUND = 116
    HARD_SWISH = 117
    IF = 118
    WHILE = 119
    NON_MAX_SUPPRESSION_V4 = 120
    NON_MAX_SUPPRESSION_V5 = 121
    SCATTER_ND = 122
    SELECT_V2 = 123
    DENSIFY = 124
    SEGMENT_SUM = 125
    BATCH_MATMUL = 126
    PLACEHOLDER_FOR_GREATER_OP_CODES = 127


class PadV2Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsPadV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PadV2Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def PadV2OptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # PadV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def PadV2OptionsStart(builder):
    builder.StartObject(0)


def PadV2OptionsEnd(builder):
    return builder.EndObject()


class PadV2OptionsT(object):

    # PadV2OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        padV2Options = PadV2Options()
        padV2Options.Init(buf, pos)
        return cls.InitFromObj(padV2Options)

    @classmethod
    def InitFromObj(cls, padV2Options):
        x = PadV2OptionsT()
        x._UnPack(padV2Options)
        return x

    # PadV2OptionsT
    def _UnPack(self, padV2Options):
        if padV2Options is None:
            return

    # PadV2OptionsT
    def Pack(self, builder):
        PadV2OptionsStart(builder)
        padV2Options = PadV2OptionsEnd(builder)
        return padV2Options


class SplitVOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSplitVOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SplitVOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SplitVOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SplitVOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SplitVOptions
    def NumSplits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def SplitVOptionsStart(builder):
    builder.StartObject(1)


def SplitVOptionsAddNumSplits(builder, numSplits):
    builder.PrependInt32Slot(0, numSplits, 0)


def SplitVOptionsEnd(builder):
    return builder.EndObject()


class SplitVOptionsT(object):

    # SplitVOptionsT
    def __init__(self):
        self.numSplits = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        splitVOptions = SplitVOptions()
        splitVOptions.Init(buf, pos)
        return cls.InitFromObj(splitVOptions)

    @classmethod
    def InitFromObj(cls, splitVOptions):
        x = SplitVOptionsT()
        x._UnPack(splitVOptions)
        return x

    # SplitVOptionsT
    def _UnPack(self, splitVOptions):
        if splitVOptions is None:
            return
        self.numSplits = splitVOptions.NumSplits()

    # SplitVOptionsT
    def Pack(self, builder):
        SplitVOptionsStart(builder)
        SplitVOptionsAddNumSplits(builder, self.numSplits)
        splitVOptions = SplitVOptionsEnd(builder)
        return splitVOptions


class Model(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsModel(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Model()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ModelBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Model
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Model
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Uint32Flags, o + self._tab.Pos
            )
        return 0

    # Model
    def OperatorCodes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = OperatorCode()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def OperatorCodesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def OperatorCodesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Model
    def Subgraphs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = SubGraph()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def SubgraphsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def SubgraphsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Model
    def Description(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Model
    def Buffers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Buffer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def BuffersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def BuffersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Model
    def MetadataBuffer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Model
    def MetadataBufferAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Model
    def MetadataBufferLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def MetadataBufferIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Model
    def Metadata(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Metadata()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def MetadataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def MetadataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0


def ModelStart(builder):
    builder.StartObject(7)


def ModelAddVersion(builder, version):
    builder.PrependUint32Slot(0, version, 0)


def ModelAddOperatorCodes(builder, operatorCodes):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(operatorCodes), 0
    )


def ModelStartOperatorCodesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ModelAddSubgraphs(builder, subgraphs):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(subgraphs), 0
    )


def ModelStartSubgraphsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ModelAddDescription(builder, description):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(description), 0
    )


def ModelAddBuffers(builder, buffers):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(buffers), 0
    )


def ModelStartBuffersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ModelAddMetadataBuffer(builder, metadataBuffer):
    builder.PrependUOffsetTRelativeSlot(
        5, flatbuffers.number_types.UOffsetTFlags.py_type(metadataBuffer), 0
    )


def ModelStartMetadataBufferVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ModelAddMetadata(builder, metadata):
    builder.PrependUOffsetTRelativeSlot(
        6, flatbuffers.number_types.UOffsetTFlags.py_type(metadata), 0
    )


def ModelStartMetadataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ModelEnd(builder):
    return builder.EndObject()


class ModelT(object):

    # ModelT
    def __init__(self):
        self.version = 0  # type: int
        self.operatorCodes = None  # type: List[OperatorCodeT]
        self.subgraphs = None  # type: List[SubGraphT]
        self.description = None  # type: str
        self.buffers = None  # type: List[BufferT]
        self.metadataBuffer = None  # type: List[int]
        self.metadata = None  # type: List[MetadataT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        model = Model()
        model.Init(buf, pos)
        return cls.InitFromObj(model)

    @classmethod
    def InitFromObj(cls, model):
        x = ModelT()
        x._UnPack(model)
        return x

    # ModelT
    def _UnPack(self, model):
        if model is None:
            return
        self.version = model.Version()
        if not model.OperatorCodesIsNone():
            self.operatorCodes = []
            for i in range(model.OperatorCodesLength()):
                if model.OperatorCodes(i) is None:
                    self.operatorCodes.append(None)
                else:
                    operatorCode_ = OperatorCodeT.InitFromObj(model.OperatorCodes(i))
                    self.operatorCodes.append(operatorCode_)
        if not model.SubgraphsIsNone():
            self.subgraphs = []
            for i in range(model.SubgraphsLength()):
                if model.Subgraphs(i) is None:
                    self.subgraphs.append(None)
                else:
                    subGraph_ = SubGraphT.InitFromObj(model.Subgraphs(i))
                    self.subgraphs.append(subGraph_)
        self.description = model.Description()
        if not model.BuffersIsNone():
            self.buffers = []
            for i in range(model.BuffersLength()):
                if model.Buffers(i) is None:
                    self.buffers.append(None)
                else:
                    buffer_ = BufferT.InitFromObj(model.Buffers(i))
                    self.buffers.append(buffer_)
        if not model.MetadataBufferIsNone():
            if np is None:
                self.metadataBuffer = []
                for i in range(model.MetadataBufferLength()):
                    self.metadataBuffer.append(model.MetadataBuffer(i))
            else:
                self.metadataBuffer = model.MetadataBufferAsNumpy()
        if not model.MetadataIsNone():
            self.metadata = []
            for i in range(model.MetadataLength()):
                if model.Metadata(i) is None:
                    self.metadata.append(None)
                else:
                    metadata_ = MetadataT.InitFromObj(model.Metadata(i))
                    self.metadata.append(metadata_)

    # ModelT
    def Pack(self, builder):
        if self.operatorCodes is not None:
            operatorCodeslist = []
            for i in range(len(self.operatorCodes)):
                operatorCodeslist.append(self.operatorCodes[i].Pack(builder))
            ModelStartOperatorCodesVector(builder, len(self.operatorCodes))
            for i in reversed(range(len(self.operatorCodes))):
                builder.PrependUOffsetTRelative(operatorCodeslist[i])
            operatorCodes = builder.EndVector(len(self.operatorCodes))
        if self.subgraphs is not None:
            subgraphslist = []
            for i in range(len(self.subgraphs)):
                subgraphslist.append(self.subgraphs[i].Pack(builder))
            ModelStartSubgraphsVector(builder, len(self.subgraphs))
            for i in reversed(range(len(self.subgraphs))):
                builder.PrependUOffsetTRelative(subgraphslist[i])
            subgraphs = builder.EndVector(len(self.subgraphs))
        if self.description is not None:
            description = builder.CreateString(self.description)
        if self.buffers is not None:
            bufferslist = []
            for i in range(len(self.buffers)):
                bufferslist.append(self.buffers[i].Pack(builder))
            ModelStartBuffersVector(builder, len(self.buffers))
            for i in reversed(range(len(self.buffers))):
                builder.PrependUOffsetTRelative(bufferslist[i])
            buffers = builder.EndVector(len(self.buffers))
        if self.metadataBuffer is not None:
            if np is not None and type(self.metadataBuffer) is np.ndarray:
                metadataBuffer = builder.CreateNumpyVector(self.metadataBuffer)
            else:
                ModelStartMetadataBufferVector(builder, len(self.metadataBuffer))
                for i in reversed(range(len(self.metadataBuffer))):
                    builder.PrependInt32(self.metadataBuffer[i])
                metadataBuffer = builder.EndVector(len(self.metadataBuffer))
        if self.metadata is not None:
            metadatalist = []
            for i in range(len(self.metadata)):
                metadatalist.append(self.metadata[i].Pack(builder))
            ModelStartMetadataVector(builder, len(self.metadata))
            for i in reversed(range(len(self.metadata))):
                builder.PrependUOffsetTRelative(metadatalist[i])
            metadata = builder.EndVector(len(self.metadata))
        ModelStart(builder)
        ModelAddVersion(builder, self.version)
        if self.operatorCodes is not None:
            ModelAddOperatorCodes(builder, operatorCodes)
        if self.subgraphs is not None:
            ModelAddSubgraphs(builder, subgraphs)
        if self.description is not None:
            ModelAddDescription(builder, description)
        if self.buffers is not None:
            ModelAddBuffers(builder, buffers)
        if self.metadataBuffer is not None:
            ModelAddMetadataBuffer(builder, metadataBuffer)
        if self.metadata is not None:
            ModelAddMetadata(builder, metadata)
        model = ModelEnd(builder)
        return model


class Tensor(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsTensor(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tensor()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def TensorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Tensor
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tensor
    def Shape(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Tensor
    def ShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Tensor
    def ShapeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def ShapeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Tensor
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Tensor
    def Buffer(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Uint32Flags, o + self._tab.Pos
            )
        return 0

    # Tensor
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tensor
    def Quantization(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = QuantizationParameters()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tensor
    def IsVariable(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # Tensor
    def Sparsity(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = SparsityParameters()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tensor
    def ShapeSignature(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # Tensor
    def ShapeSignatureAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Tensor
    def ShapeSignatureLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def ShapeSignatureIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0


def TensorStart(builder):
    builder.StartObject(8)


def TensorAddShape(builder, shape):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0
    )


def TensorStartShapeVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def TensorAddType(builder, type):
    builder.PrependInt8Slot(1, type, 0)


def TensorAddBuffer(builder, buffer):
    builder.PrependUint32Slot(2, buffer, 0)


def TensorAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0
    )


def TensorAddQuantization(builder, quantization):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(quantization), 0
    )


def TensorAddIsVariable(builder, isVariable):
    builder.PrependBoolSlot(5, isVariable, 0)


def TensorAddSparsity(builder, sparsity):
    builder.PrependUOffsetTRelativeSlot(
        6, flatbuffers.number_types.UOffsetTFlags.py_type(sparsity), 0
    )


def TensorAddShapeSignature(builder, shapeSignature):
    builder.PrependUOffsetTRelativeSlot(
        7, flatbuffers.number_types.UOffsetTFlags.py_type(shapeSignature), 0
    )


def TensorStartShapeSignatureVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def TensorEnd(builder):
    return builder.EndObject()


class TensorT(object):

    # TensorT
    def __init__(self):
        self.shape = None  # type: List[int]
        self.type = 0  # type: int
        self.buffer = 0  # type: int
        self.name = None  # type: str
        self.quantization = None  # type: Optional[QuantizationParametersT]
        self.isVariable = False  # type: bool
        self.sparsity = None  # type: Optional[SparsityParametersT]
        self.shapeSignature = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tensor = Tensor()
        tensor.Init(buf, pos)
        return cls.InitFromObj(tensor)

    @classmethod
    def InitFromObj(cls, tensor):
        x = TensorT()
        x._UnPack(tensor)
        return x

    # TensorT
    def _UnPack(self, tensor):
        if tensor is None:
            return
        if not tensor.ShapeIsNone():
            if np is None:
                self.shape = []
                for i in range(tensor.ShapeLength()):
                    self.shape.append(tensor.Shape(i))
            else:
                self.shape = tensor.ShapeAsNumpy()
        self.type = tensor.Type()
        self.buffer = tensor.Buffer()
        self.name = tensor.Name()
        if tensor.Quantization() is not None:
            self.quantization = QuantizationParametersT.InitFromObj(
                tensor.Quantization()
            )
        self.isVariable = tensor.IsVariable()
        if tensor.Sparsity() is not None:
            self.sparsity = SparsityParametersT.InitFromObj(tensor.Sparsity())
        if not tensor.ShapeSignatureIsNone():
            if np is None:
                self.shapeSignature = []
                for i in range(tensor.ShapeSignatureLength()):
                    self.shapeSignature.append(tensor.ShapeSignature(i))
            else:
                self.shapeSignature = tensor.ShapeSignatureAsNumpy()

    # TensorT
    def Pack(self, builder):
        if self.shape is not None:
            if np is not None and type(self.shape) is np.ndarray:
                shape = builder.CreateNumpyVector(self.shape)
            else:
                TensorStartShapeVector(builder, len(self.shape))
                for i in reversed(range(len(self.shape))):
                    builder.PrependInt32(self.shape[i])
                shape = builder.EndVector(len(self.shape))
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.quantization is not None:
            quantization = self.quantization.Pack(builder)
        if self.sparsity is not None:
            sparsity = self.sparsity.Pack(builder)
        if self.shapeSignature is not None:
            if np is not None and type(self.shapeSignature) is np.ndarray:
                shapeSignature = builder.CreateNumpyVector(self.shapeSignature)
            else:
                TensorStartShapeSignatureVector(builder, len(self.shapeSignature))
                for i in reversed(range(len(self.shapeSignature))):
                    builder.PrependInt32(self.shapeSignature[i])
                shapeSignature = builder.EndVector(len(self.shapeSignature))
        TensorStart(builder)
        if self.shape is not None:
            TensorAddShape(builder, shape)
        TensorAddType(builder, self.type)
        TensorAddBuffer(builder, self.buffer)
        if self.name is not None:
            TensorAddName(builder, name)
        if self.quantization is not None:
            TensorAddQuantization(builder, quantization)
        TensorAddIsVariable(builder, self.isVariable)
        if self.sparsity is not None:
            TensorAddSparsity(builder, sparsity)
        if self.shapeSignature is not None:
            TensorAddShapeSignature(builder, shapeSignature)
        tensor = TensorEnd(builder)
        return tensor


class SkipGramOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSkipGramOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SkipGramOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SkipGramOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SkipGramOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SkipGramOptions
    def NgramSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # SkipGramOptions
    def MaxSkipSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # SkipGramOptions
    def IncludeAllNgrams(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def SkipGramOptionsStart(builder):
    builder.StartObject(3)


def SkipGramOptionsAddNgramSize(builder, ngramSize):
    builder.PrependInt32Slot(0, ngramSize, 0)


def SkipGramOptionsAddMaxSkipSize(builder, maxSkipSize):
    builder.PrependInt32Slot(1, maxSkipSize, 0)


def SkipGramOptionsAddIncludeAllNgrams(builder, includeAllNgrams):
    builder.PrependBoolSlot(2, includeAllNgrams, 0)


def SkipGramOptionsEnd(builder):
    return builder.EndObject()


class SkipGramOptionsT(object):

    # SkipGramOptionsT
    def __init__(self):
        self.ngramSize = 0  # type: int
        self.maxSkipSize = 0  # type: int
        self.includeAllNgrams = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        skipGramOptions = SkipGramOptions()
        skipGramOptions.Init(buf, pos)
        return cls.InitFromObj(skipGramOptions)

    @classmethod
    def InitFromObj(cls, skipGramOptions):
        x = SkipGramOptionsT()
        x._UnPack(skipGramOptions)
        return x

    # SkipGramOptionsT
    def _UnPack(self, skipGramOptions):
        if skipGramOptions is None:
            return
        self.ngramSize = skipGramOptions.NgramSize()
        self.maxSkipSize = skipGramOptions.MaxSkipSize()
        self.includeAllNgrams = skipGramOptions.IncludeAllNgrams()

    # SkipGramOptionsT
    def Pack(self, builder):
        SkipGramOptionsStart(builder)
        SkipGramOptionsAddNgramSize(builder, self.ngramSize)
        SkipGramOptionsAddMaxSkipSize(builder, self.maxSkipSize)
        SkipGramOptionsAddIncludeAllNgrams(builder, self.includeAllNgrams)
        skipGramOptions = SkipGramOptionsEnd(builder)
        return skipGramOptions


class LogicalNotOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLogicalNotOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalNotOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LogicalNotOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LogicalNotOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LogicalNotOptionsStart(builder):
    builder.StartObject(0)


def LogicalNotOptionsEnd(builder):
    return builder.EndObject()


class LogicalNotOptionsT(object):

    # LogicalNotOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalNotOptions = LogicalNotOptions()
        logicalNotOptions.Init(buf, pos)
        return cls.InitFromObj(logicalNotOptions)

    @classmethod
    def InitFromObj(cls, logicalNotOptions):
        x = LogicalNotOptionsT()
        x._UnPack(logicalNotOptions)
        return x

    # LogicalNotOptionsT
    def _UnPack(self, logicalNotOptions):
        if logicalNotOptions is None:
            return

    # LogicalNotOptionsT
    def Pack(self, builder):
        LogicalNotOptionsStart(builder)
        logicalNotOptions = LogicalNotOptionsEnd(builder)
        return logicalNotOptions


class CosOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsCosOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CosOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CosOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # CosOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def CosOptionsStart(builder):
    builder.StartObject(0)


def CosOptionsEnd(builder):
    return builder.EndObject()


class CosOptionsT(object):

    # CosOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        cosOptions = CosOptions()
        cosOptions.Init(buf, pos)
        return cls.InitFromObj(cosOptions)

    @classmethod
    def InitFromObj(cls, cosOptions):
        x = CosOptionsT()
        x._UnPack(cosOptions)
        return x

    # CosOptionsT
    def _UnPack(self, cosOptions):
        if cosOptions is None:
            return

    # CosOptionsT
    def Pack(self, builder):
        CosOptionsStart(builder)
        cosOptions = CosOptionsEnd(builder)
        return cosOptions


class HardSwishOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsHardSwishOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HardSwishOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def HardSwishOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # HardSwishOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def HardSwishOptionsStart(builder):
    builder.StartObject(0)


def HardSwishOptionsEnd(builder):
    return builder.EndObject()


class HardSwishOptionsT(object):

    # HardSwishOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        hardSwishOptions = HardSwishOptions()
        hardSwishOptions.Init(buf, pos)
        return cls.InitFromObj(hardSwishOptions)

    @classmethod
    def InitFromObj(cls, hardSwishOptions):
        x = HardSwishOptionsT()
        x._UnPack(hardSwishOptions)
        return x

    # HardSwishOptionsT
    def _UnPack(self, hardSwishOptions):
        if hardSwishOptions is None:
            return

    # HardSwishOptionsT
    def Pack(self, builder):
        HardSwishOptionsStart(builder)
        hardSwishOptions = HardSwishOptionsEnd(builder)
        return hardSwishOptions


class ExpOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsExpOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ExpOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ExpOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ExpOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ExpOptionsStart(builder):
    builder.StartObject(0)


def ExpOptionsEnd(builder):
    return builder.EndObject()


class ExpOptionsT(object):

    # ExpOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        expOptions = ExpOptions()
        expOptions.Init(buf, pos)
        return cls.InitFromObj(expOptions)

    @classmethod
    def InitFromObj(cls, expOptions):
        x = ExpOptionsT()
        x._UnPack(expOptions)
        return x

    # ExpOptionsT
    def _UnPack(self, expOptions):
        if expOptions is None:
            return

    # ExpOptionsT
    def Pack(self, builder):
        ExpOptionsStart(builder)
        expOptions = ExpOptionsEnd(builder)
        return expOptions


class MulOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMulOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MulOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MulOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # MulOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MulOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def MulOptionsStart(builder):
    builder.StartObject(1)


def MulOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def MulOptionsEnd(builder):
    return builder.EndObject()


class MulOptionsT(object):

    # MulOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        mulOptions = MulOptions()
        mulOptions.Init(buf, pos)
        return cls.InitFromObj(mulOptions)

    @classmethod
    def InitFromObj(cls, mulOptions):
        x = MulOptionsT()
        x._UnPack(mulOptions)
        return x

    # MulOptionsT
    def _UnPack(self, mulOptions):
        if mulOptions is None:
            return
        self.fusedActivationFunction = mulOptions.FusedActivationFunction()

    # MulOptionsT
    def Pack(self, builder):
        MulOptionsStart(builder)
        MulOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        mulOptions = MulOptionsEnd(builder)
        return mulOptions


class FillOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsFillOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FillOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FillOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # FillOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def FillOptionsStart(builder):
    builder.StartObject(0)


def FillOptionsEnd(builder):
    return builder.EndObject()


class FillOptionsT(object):

    # FillOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fillOptions = FillOptions()
        fillOptions.Init(buf, pos)
        return cls.InitFromObj(fillOptions)

    @classmethod
    def InitFromObj(cls, fillOptions):
        x = FillOptionsT()
        x._UnPack(fillOptions)
        return x

    # FillOptionsT
    def _UnPack(self, fillOptions):
        if fillOptions is None:
            return

    # FillOptionsT
    def Pack(self, builder):
        FillOptionsStart(builder)
        fillOptions = FillOptionsEnd(builder)
        return fillOptions


class BidirectionalSequenceRNNOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsBidirectionalSequenceRNNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BidirectionalSequenceRNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BidirectionalSequenceRNNOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # BidirectionalSequenceRNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BidirectionalSequenceRNNOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # BidirectionalSequenceRNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # BidirectionalSequenceRNNOptions
    def MergeOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # BidirectionalSequenceRNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def BidirectionalSequenceRNNOptionsStart(builder):
    builder.StartObject(4)


def BidirectionalSequenceRNNOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(0, timeMajor, 0)


def BidirectionalSequenceRNNOptionsAddFusedActivationFunction(
    builder, fusedActivationFunction
):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def BidirectionalSequenceRNNOptionsAddMergeOutputs(builder, mergeOutputs):
    builder.PrependBoolSlot(2, mergeOutputs, 0)


def BidirectionalSequenceRNNOptionsAddAsymmetricQuantizeInputs(
    builder, asymmetricQuantizeInputs
):
    builder.PrependBoolSlot(3, asymmetricQuantizeInputs, 0)


def BidirectionalSequenceRNNOptionsEnd(builder):
    return builder.EndObject()


class BidirectionalSequenceRNNOptionsT(object):

    # BidirectionalSequenceRNNOptionsT
    def __init__(self):
        self.timeMajor = False  # type: bool
        self.fusedActivationFunction = 0  # type: int
        self.mergeOutputs = False  # type: bool
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bidirectionalSequenceRNNOptions = BidirectionalSequenceRNNOptions()
        bidirectionalSequenceRNNOptions.Init(buf, pos)
        return cls.InitFromObj(bidirectionalSequenceRNNOptions)

    @classmethod
    def InitFromObj(cls, bidirectionalSequenceRNNOptions):
        x = BidirectionalSequenceRNNOptionsT()
        x._UnPack(bidirectionalSequenceRNNOptions)
        return x

    # BidirectionalSequenceRNNOptionsT
    def _UnPack(self, bidirectionalSequenceRNNOptions):
        if bidirectionalSequenceRNNOptions is None:
            return
        self.timeMajor = bidirectionalSequenceRNNOptions.TimeMajor()
        self.fusedActivationFunction = (
            bidirectionalSequenceRNNOptions.FusedActivationFunction()
        )
        self.mergeOutputs = bidirectionalSequenceRNNOptions.MergeOutputs()
        self.asymmetricQuantizeInputs = (
            bidirectionalSequenceRNNOptions.AsymmetricQuantizeInputs()
        )

    # BidirectionalSequenceRNNOptionsT
    def Pack(self, builder):
        BidirectionalSequenceRNNOptionsStart(builder)
        BidirectionalSequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        BidirectionalSequenceRNNOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        BidirectionalSequenceRNNOptionsAddMergeOutputs(builder, self.mergeOutputs)
        BidirectionalSequenceRNNOptionsAddAsymmetricQuantizeInputs(
            builder, self.asymmetricQuantizeInputs
        )
        bidirectionalSequenceRNNOptions = BidirectionalSequenceRNNOptionsEnd(builder)
        return bidirectionalSequenceRNNOptions


class FloorModOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsFloorModOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloorModOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FloorModOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # FloorModOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def FloorModOptionsStart(builder):
    builder.StartObject(0)


def FloorModOptionsEnd(builder):
    return builder.EndObject()


class FloorModOptionsT(object):

    # FloorModOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        floorModOptions = FloorModOptions()
        floorModOptions.Init(buf, pos)
        return cls.InitFromObj(floorModOptions)

    @classmethod
    def InitFromObj(cls, floorModOptions):
        x = FloorModOptionsT()
        x._UnPack(floorModOptions)
        return x

    # FloorModOptionsT
    def _UnPack(self, floorModOptions):
        if floorModOptions is None:
            return

    # FloorModOptionsT
    def Pack(self, builder):
        FloorModOptionsStart(builder)
        floorModOptions = FloorModOptionsEnd(builder)
        return floorModOptions


class ScatterNdOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsScatterNdOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ScatterNdOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ScatterNdOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ScatterNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ScatterNdOptionsStart(builder):
    builder.StartObject(0)


def ScatterNdOptionsEnd(builder):
    return builder.EndObject()


class ScatterNdOptionsT(object):

    # ScatterNdOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        scatterNdOptions = ScatterNdOptions()
        scatterNdOptions.Init(buf, pos)
        return cls.InitFromObj(scatterNdOptions)

    @classmethod
    def InitFromObj(cls, scatterNdOptions):
        x = ScatterNdOptionsT()
        x._UnPack(scatterNdOptions)
        return x

    # ScatterNdOptionsT
    def _UnPack(self, scatterNdOptions):
        if scatterNdOptions is None:
            return

    # ScatterNdOptionsT
    def Pack(self, builder):
        ScatterNdOptionsStart(builder)
        scatterNdOptions = ScatterNdOptionsEnd(builder)
        return scatterNdOptions


class LogicalAndOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLogicalAndOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalAndOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LogicalAndOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LogicalAndOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def LogicalAndOptionsStart(builder):
    builder.StartObject(0)


def LogicalAndOptionsEnd(builder):
    return builder.EndObject()


class LogicalAndOptionsT(object):

    # LogicalAndOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        logicalAndOptions = LogicalAndOptions()
        logicalAndOptions.Init(buf, pos)
        return cls.InitFromObj(logicalAndOptions)

    @classmethod
    def InitFromObj(cls, logicalAndOptions):
        x = LogicalAndOptionsT()
        x._UnPack(logicalAndOptions)
        return x

    # LogicalAndOptionsT
    def _UnPack(self, logicalAndOptions):
        if logicalAndOptions is None:
            return

    # LogicalAndOptionsT
    def Pack(self, builder):
        LogicalAndOptionsStart(builder)
        logicalAndOptions = LogicalAndOptionsEnd(builder)
        return logicalAndOptions


class SqueezeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSqueezeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SqueezeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SqueezeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SqueezeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SqueezeOptions
    def SqueezeDims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SqueezeOptions
    def SqueezeDimsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SqueezeOptions
    def SqueezeDimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SqueezeOptions
    def SqueezeDimsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def SqueezeOptionsStart(builder):
    builder.StartObject(1)


def SqueezeOptionsAddSqueezeDims(builder, squeezeDims):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(squeezeDims), 0
    )


def SqueezeOptionsStartSqueezeDimsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SqueezeOptionsEnd(builder):
    return builder.EndObject()


class SqueezeOptionsT(object):

    # SqueezeOptionsT
    def __init__(self):
        self.squeezeDims = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squeezeOptions = SqueezeOptions()
        squeezeOptions.Init(buf, pos)
        return cls.InitFromObj(squeezeOptions)

    @classmethod
    def InitFromObj(cls, squeezeOptions):
        x = SqueezeOptionsT()
        x._UnPack(squeezeOptions)
        return x

    # SqueezeOptionsT
    def _UnPack(self, squeezeOptions):
        if squeezeOptions is None:
            return
        if not squeezeOptions.SqueezeDimsIsNone():
            if np is None:
                self.squeezeDims = []
                for i in range(squeezeOptions.SqueezeDimsLength()):
                    self.squeezeDims.append(squeezeOptions.SqueezeDims(i))
            else:
                self.squeezeDims = squeezeOptions.SqueezeDimsAsNumpy()

    # SqueezeOptionsT
    def Pack(self, builder):
        if self.squeezeDims is not None:
            if np is not None and type(self.squeezeDims) is np.ndarray:
                squeezeDims = builder.CreateNumpyVector(self.squeezeDims)
            else:
                SqueezeOptionsStartSqueezeDimsVector(builder, len(self.squeezeDims))
                for i in reversed(range(len(self.squeezeDims))):
                    builder.PrependInt32(self.squeezeDims[i])
                squeezeDims = builder.EndVector(len(self.squeezeDims))
        SqueezeOptionsStart(builder)
        if self.squeezeDims is not None:
            SqueezeOptionsAddSqueezeDims(builder, squeezeDims)
        squeezeOptions = SqueezeOptionsEnd(builder)
        return squeezeOptions


class SequenceRNNOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSequenceRNNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SequenceRNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SequenceRNNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SequenceRNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SequenceRNNOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # SequenceRNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # SequenceRNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def SequenceRNNOptionsStart(builder):
    builder.StartObject(3)


def SequenceRNNOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(0, timeMajor, 0)


def SequenceRNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)


def SequenceRNNOptionsEnd(builder):
    return builder.EndObject()


class SequenceRNNOptionsT(object):

    # SequenceRNNOptionsT
    def __init__(self):
        self.timeMajor = False  # type: bool
        self.fusedActivationFunction = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sequenceRNNOptions = SequenceRNNOptions()
        sequenceRNNOptions.Init(buf, pos)
        return cls.InitFromObj(sequenceRNNOptions)

    @classmethod
    def InitFromObj(cls, sequenceRNNOptions):
        x = SequenceRNNOptionsT()
        x._UnPack(sequenceRNNOptions)
        return x

    # SequenceRNNOptionsT
    def _UnPack(self, sequenceRNNOptions):
        if sequenceRNNOptions is None:
            return
        self.timeMajor = sequenceRNNOptions.TimeMajor()
        self.fusedActivationFunction = sequenceRNNOptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = sequenceRNNOptions.AsymmetricQuantizeInputs()

    # SequenceRNNOptionsT
    def Pack(self, builder):
        SequenceRNNOptionsStart(builder)
        SequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        SequenceRNNOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        SequenceRNNOptionsAddAsymmetricQuantizeInputs(
            builder, self.asymmetricQuantizeInputs
        )
        sequenceRNNOptions = SequenceRNNOptionsEnd(builder)
        return sequenceRNNOptions


class SquaredDifferenceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSquaredDifferenceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SquaredDifferenceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SquaredDifferenceOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SquaredDifferenceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SquaredDifferenceOptionsStart(builder):
    builder.StartObject(0)


def SquaredDifferenceOptionsEnd(builder):
    return builder.EndObject()


class SquaredDifferenceOptionsT(object):

    # SquaredDifferenceOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squaredDifferenceOptions = SquaredDifferenceOptions()
        squaredDifferenceOptions.Init(buf, pos)
        return cls.InitFromObj(squaredDifferenceOptions)

    @classmethod
    def InitFromObj(cls, squaredDifferenceOptions):
        x = SquaredDifferenceOptionsT()
        x._UnPack(squaredDifferenceOptions)
        return x

    # SquaredDifferenceOptionsT
    def _UnPack(self, squaredDifferenceOptions):
        if squaredDifferenceOptions is None:
            return

    # SquaredDifferenceOptionsT
    def Pack(self, builder):
        SquaredDifferenceOptionsStart(builder)
        squaredDifferenceOptions = SquaredDifferenceOptionsEnd(builder)
        return squaredDifferenceOptions


class QuantizationDetails(object):
    NONE = 0
    CustomQuantization = 1


def QuantizationDetailsCreator(unionType, table):
    from flatbuffers.table import Table

    if not isinstance(table, Table):
        return None
    if unionType == QuantizationDetails().CustomQuantization:
        return CustomQuantizationT.InitFromBuf(table.Bytes, table.Pos)
    return None


class SpaceToBatchNDOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSpaceToBatchNDOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SpaceToBatchNDOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SpaceToBatchNDOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SpaceToBatchNDOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SpaceToBatchNDOptionsStart(builder):
    builder.StartObject(0)


def SpaceToBatchNDOptionsEnd(builder):
    return builder.EndObject()


class SpaceToBatchNDOptionsT(object):

    # SpaceToBatchNDOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        spaceToBatchNDOptions = SpaceToBatchNDOptions()
        spaceToBatchNDOptions.Init(buf, pos)
        return cls.InitFromObj(spaceToBatchNDOptions)

    @classmethod
    def InitFromObj(cls, spaceToBatchNDOptions):
        x = SpaceToBatchNDOptionsT()
        x._UnPack(spaceToBatchNDOptions)
        return x

    # SpaceToBatchNDOptionsT
    def _UnPack(self, spaceToBatchNDOptions):
        if spaceToBatchNDOptions is None:
            return

    # SpaceToBatchNDOptionsT
    def Pack(self, builder):
        SpaceToBatchNDOptionsStart(builder)
        spaceToBatchNDOptions = SpaceToBatchNDOptionsEnd(builder)
        return spaceToBatchNDOptions


class OperatorCode(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsOperatorCode(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OperatorCode()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def OperatorCodeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # OperatorCode
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OperatorCode
    def DeprecatedBuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # OperatorCode
    def CustomCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OperatorCode
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # OperatorCode
    def BuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def OperatorCodeStart(builder):
    builder.StartObject(4)


def OperatorCodeAddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode):
    builder.PrependInt8Slot(0, deprecatedBuiltinCode, 0)


def OperatorCodeAddCustomCode(builder, customCode):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(customCode), 0
    )


def OperatorCodeAddVersion(builder, version):
    builder.PrependInt32Slot(2, version, 1)


def OperatorCodeAddBuiltinCode(builder, builtinCode):
    builder.PrependInt32Slot(3, builtinCode, 0)


def OperatorCodeEnd(builder):
    return builder.EndObject()


class OperatorCodeT(object):

    # OperatorCodeT
    def __init__(self):
        self.deprecatedBuiltinCode = 0  # type: int
        self.customCode = None  # type: str
        self.version = 1  # type: int
        self.builtinCode = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operatorCode = OperatorCode()
        operatorCode.Init(buf, pos)
        return cls.InitFromObj(operatorCode)

    @classmethod
    def InitFromObj(cls, operatorCode):
        x = OperatorCodeT()
        x._UnPack(operatorCode)
        return x

    # OperatorCodeT
    def _UnPack(self, operatorCode):
        if operatorCode is None:
            return
        self.deprecatedBuiltinCode = operatorCode.DeprecatedBuiltinCode()
        self.customCode = operatorCode.CustomCode()
        self.version = operatorCode.Version()
        self.builtinCode = operatorCode.BuiltinCode()

    # OperatorCodeT
    def Pack(self, builder):
        if self.customCode is not None:
            customCode = builder.CreateString(self.customCode)
        OperatorCodeStart(builder)
        OperatorCodeAddDeprecatedBuiltinCode(builder, self.deprecatedBuiltinCode)
        if self.customCode is not None:
            OperatorCodeAddCustomCode(builder, customCode)
        OperatorCodeAddVersion(builder, self.version)
        OperatorCodeAddBuiltinCode(builder, self.builtinCode)
        operatorCode = OperatorCodeEnd(builder)
        return operatorCode


class LSHProjectionType(object):
    UNKNOWN = 0
    SPARSE = 1
    DENSE = 2


class SparseToDenseOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSparseToDenseOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SparseToDenseOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SparseToDenseOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SparseToDenseOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SparseToDenseOptions
    def ValidateIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def SparseToDenseOptionsStart(builder):
    builder.StartObject(1)


def SparseToDenseOptionsAddValidateIndices(builder, validateIndices):
    builder.PrependBoolSlot(0, validateIndices, 0)


def SparseToDenseOptionsEnd(builder):
    return builder.EndObject()


class SparseToDenseOptionsT(object):

    # SparseToDenseOptionsT
    def __init__(self):
        self.validateIndices = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparseToDenseOptions = SparseToDenseOptions()
        sparseToDenseOptions.Init(buf, pos)
        return cls.InitFromObj(sparseToDenseOptions)

    @classmethod
    def InitFromObj(cls, sparseToDenseOptions):
        x = SparseToDenseOptionsT()
        x._UnPack(sparseToDenseOptions)
        return x

    # SparseToDenseOptionsT
    def _UnPack(self, sparseToDenseOptions):
        if sparseToDenseOptions is None:
            return
        self.validateIndices = sparseToDenseOptions.ValidateIndices()

    # SparseToDenseOptionsT
    def Pack(self, builder):
        SparseToDenseOptionsStart(builder)
        SparseToDenseOptionsAddValidateIndices(builder, self.validateIndices)
        sparseToDenseOptions = SparseToDenseOptionsEnd(builder)
        return sparseToDenseOptions


class RangeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsRangeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RangeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def RangeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # RangeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def RangeOptionsStart(builder):
    builder.StartObject(0)


def RangeOptionsEnd(builder):
    return builder.EndObject()


class RangeOptionsT(object):

    # RangeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rangeOptions = RangeOptions()
        rangeOptions.Init(buf, pos)
        return cls.InitFromObj(rangeOptions)

    @classmethod
    def InitFromObj(cls, rangeOptions):
        x = RangeOptionsT()
        x._UnPack(rangeOptions)
        return x

    # RangeOptionsT
    def _UnPack(self, rangeOptions):
        if rangeOptions is None:
            return

    # RangeOptionsT
    def Pack(self, builder):
        RangeOptionsStart(builder)
        rangeOptions = RangeOptionsEnd(builder)
        return rangeOptions


class Buffer(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsBuffer(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Buffer()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BufferBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Buffer
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Buffer
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Uint8Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1),
            )
        return 0

    # Buffer
    def DataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Buffer
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Buffer
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def BufferStart(builder):
    builder.StartObject(1)


def BufferAddData(builder, data):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0
    )


def BufferStartDataVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)


def BufferEnd(builder):
    return builder.EndObject()


class BufferT(object):

    # BufferT
    def __init__(self):
        self.data = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        buffer = Buffer()
        buffer.Init(buf, pos)
        return cls.InitFromObj(buffer)

    @classmethod
    def InitFromObj(cls, buffer):
        x = BufferT()
        x._UnPack(buffer)
        return x

    # BufferT
    def _UnPack(self, buffer):
        if buffer is None:
            return
        if not buffer.DataIsNone():
            if np is None:
                self.data = []
                for i in range(buffer.DataLength()):
                    self.data.append(buffer.Data(i))
            else:
                self.data = buffer.DataAsNumpy()

    # BufferT
    def Pack(self, builder):
        if self.data is not None:
            if np is not None and type(self.data) is np.ndarray:
                data = builder.CreateNumpyVector(self.data)
            else:
                BufferStartDataVector(builder, len(self.data))
                for i in reversed(range(len(self.data))):
                    builder.PrependUint8(self.data[i])
                data = builder.EndVector(len(self.data))
        BufferStart(builder)
        if self.data is not None:
            BufferAddData(builder, data)
        buffer = BufferEnd(builder)
        return buffer


class OneHotOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsOneHotOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OneHotOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def OneHotOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # OneHotOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OneHotOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def OneHotOptionsStart(builder):
    builder.StartObject(1)


def OneHotOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)


def OneHotOptionsEnd(builder):
    return builder.EndObject()


class OneHotOptionsT(object):

    # OneHotOptionsT
    def __init__(self):
        self.axis = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        oneHotOptions = OneHotOptions()
        oneHotOptions.Init(buf, pos)
        return cls.InitFromObj(oneHotOptions)

    @classmethod
    def InitFromObj(cls, oneHotOptions):
        x = OneHotOptionsT()
        x._UnPack(oneHotOptions)
        return x

    # OneHotOptionsT
    def _UnPack(self, oneHotOptions):
        if oneHotOptions is None:
            return
        self.axis = oneHotOptions.Axis()

    # OneHotOptionsT
    def Pack(self, builder):
        OneHotOptionsStart(builder)
        OneHotOptionsAddAxis(builder, self.axis)
        oneHotOptions = OneHotOptionsEnd(builder)
        return oneHotOptions


class DensifyOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDensifyOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DensifyOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DensifyOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DensifyOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def DensifyOptionsStart(builder):
    builder.StartObject(0)


def DensifyOptionsEnd(builder):
    return builder.EndObject()


class DensifyOptionsT(object):

    # DensifyOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        densifyOptions = DensifyOptions()
        densifyOptions.Init(buf, pos)
        return cls.InitFromObj(densifyOptions)

    @classmethod
    def InitFromObj(cls, densifyOptions):
        x = DensifyOptionsT()
        x._UnPack(densifyOptions)
        return x

    # DensifyOptionsT
    def _UnPack(self, densifyOptions):
        if densifyOptions is None:
            return

    # DensifyOptionsT
    def Pack(self, builder):
        DensifyOptionsStart(builder)
        densifyOptions = DensifyOptionsEnd(builder)
        return densifyOptions


class SVDFOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSVDFOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SVDFOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SVDFOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SVDFOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SVDFOptions
    def Rank(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # SVDFOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # SVDFOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def SVDFOptionsStart(builder):
    builder.StartObject(3)


def SVDFOptionsAddRank(builder, rank):
    builder.PrependInt32Slot(0, rank, 0)


def SVDFOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def SVDFOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)


def SVDFOptionsEnd(builder):
    return builder.EndObject()


class SVDFOptionsT(object):

    # SVDFOptionsT
    def __init__(self):
        self.rank = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sVDFOptions = SVDFOptions()
        sVDFOptions.Init(buf, pos)
        return cls.InitFromObj(sVDFOptions)

    @classmethod
    def InitFromObj(cls, sVDFOptions):
        x = SVDFOptionsT()
        x._UnPack(sVDFOptions)
        return x

    # SVDFOptionsT
    def _UnPack(self, sVDFOptions):
        if sVDFOptions is None:
            return
        self.rank = sVDFOptions.Rank()
        self.fusedActivationFunction = sVDFOptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = sVDFOptions.AsymmetricQuantizeInputs()

    # SVDFOptionsT
    def Pack(self, builder):
        SVDFOptionsStart(builder)
        SVDFOptionsAddRank(builder, self.rank)
        SVDFOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SVDFOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        sVDFOptions = SVDFOptionsEnd(builder)
        return sVDFOptions


class ConcatenationOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsConcatenationOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConcatenationOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ConcatenationOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ConcatenationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConcatenationOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConcatenationOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ConcatenationOptionsStart(builder):
    builder.StartObject(2)


def ConcatenationOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)


def ConcatenationOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def ConcatenationOptionsEnd(builder):
    return builder.EndObject()


class ConcatenationOptionsT(object):

    # ConcatenationOptionsT
    def __init__(self):
        self.axis = 0  # type: int
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        concatenationOptions = ConcatenationOptions()
        concatenationOptions.Init(buf, pos)
        return cls.InitFromObj(concatenationOptions)

    @classmethod
    def InitFromObj(cls, concatenationOptions):
        x = ConcatenationOptionsT()
        x._UnPack(concatenationOptions)
        return x

    # ConcatenationOptionsT
    def _UnPack(self, concatenationOptions):
        if concatenationOptions is None:
            return
        self.axis = concatenationOptions.Axis()
        self.fusedActivationFunction = concatenationOptions.FusedActivationFunction()

    # ConcatenationOptionsT
    def Pack(self, builder):
        ConcatenationOptionsStart(builder)
        ConcatenationOptionsAddAxis(builder, self.axis)
        ConcatenationOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        concatenationOptions = ConcatenationOptionsEnd(builder)
        return concatenationOptions


class PackOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsPackOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PackOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def PackOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # PackOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PackOptions
    def ValuesCount(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # PackOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def PackOptionsStart(builder):
    builder.StartObject(2)


def PackOptionsAddValuesCount(builder, valuesCount):
    builder.PrependInt32Slot(0, valuesCount, 0)


def PackOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(1, axis, 0)


def PackOptionsEnd(builder):
    return builder.EndObject()


class PackOptionsT(object):

    # PackOptionsT
    def __init__(self):
        self.valuesCount = 0  # type: int
        self.axis = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        packOptions = PackOptions()
        packOptions.Init(buf, pos)
        return cls.InitFromObj(packOptions)

    @classmethod
    def InitFromObj(cls, packOptions):
        x = PackOptionsT()
        x._UnPack(packOptions)
        return x

    # PackOptionsT
    def _UnPack(self, packOptions):
        if packOptions is None:
            return
        self.valuesCount = packOptions.ValuesCount()
        self.axis = packOptions.Axis()

    # PackOptionsT
    def Pack(self, builder):
        PackOptionsStart(builder)
        PackOptionsAddValuesCount(builder, self.valuesCount)
        PackOptionsAddAxis(builder, self.axis)
        packOptions = PackOptionsEnd(builder)
        return packOptions


class MatrixSetDiagOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMatrixSetDiagOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixSetDiagOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MatrixSetDiagOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # MatrixSetDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def MatrixSetDiagOptionsStart(builder):
    builder.StartObject(0)


def MatrixSetDiagOptionsEnd(builder):
    return builder.EndObject()


class MatrixSetDiagOptionsT(object):

    # MatrixSetDiagOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        matrixSetDiagOptions = MatrixSetDiagOptions()
        matrixSetDiagOptions.Init(buf, pos)
        return cls.InitFromObj(matrixSetDiagOptions)

    @classmethod
    def InitFromObj(cls, matrixSetDiagOptions):
        x = MatrixSetDiagOptionsT()
        x._UnPack(matrixSetDiagOptions)
        return x

    # MatrixSetDiagOptionsT
    def _UnPack(self, matrixSetDiagOptions):
        if matrixSetDiagOptions is None:
            return

    # MatrixSetDiagOptionsT
    def Pack(self, builder):
        MatrixSetDiagOptionsStart(builder)
        matrixSetDiagOptions = MatrixSetDiagOptionsEnd(builder)
        return matrixSetDiagOptions


class ArgMaxOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsArgMaxOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArgMaxOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ArgMaxOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ArgMaxOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArgMaxOptions
    def OutputType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def ArgMaxOptionsStart(builder):
    builder.StartObject(1)


def ArgMaxOptionsAddOutputType(builder, outputType):
    builder.PrependInt8Slot(0, outputType, 0)


def ArgMaxOptionsEnd(builder):
    return builder.EndObject()


class ArgMaxOptionsT(object):

    # ArgMaxOptionsT
    def __init__(self):
        self.outputType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        argMaxOptions = ArgMaxOptions()
        argMaxOptions.Init(buf, pos)
        return cls.InitFromObj(argMaxOptions)

    @classmethod
    def InitFromObj(cls, argMaxOptions):
        x = ArgMaxOptionsT()
        x._UnPack(argMaxOptions)
        return x

    # ArgMaxOptionsT
    def _UnPack(self, argMaxOptions):
        if argMaxOptions is None:
            return
        self.outputType = argMaxOptions.OutputType()

    # ArgMaxOptionsT
    def Pack(self, builder):
        ArgMaxOptionsStart(builder)
        ArgMaxOptionsAddOutputType(builder, self.outputType)
        argMaxOptions = ArgMaxOptionsEnd(builder)
        return argMaxOptions


class AddNOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsAddNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AddNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def AddNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # AddNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def AddNOptionsStart(builder):
    builder.StartObject(0)


def AddNOptionsEnd(builder):
    return builder.EndObject()


class AddNOptionsT(object):

    # AddNOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        addNOptions = AddNOptions()
        addNOptions.Init(buf, pos)
        return cls.InitFromObj(addNOptions)

    @classmethod
    def InitFromObj(cls, addNOptions):
        x = AddNOptionsT()
        x._UnPack(addNOptions)
        return x

    # AddNOptionsT
    def _UnPack(self, addNOptions):
        if addNOptions is None:
            return

    # AddNOptionsT
    def Pack(self, builder):
        AddNOptionsStart(builder)
        addNOptions = AddNOptionsEnd(builder)
        return addNOptions


class BatchToSpaceNDOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsBatchToSpaceNDOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BatchToSpaceNDOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BatchToSpaceNDOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # BatchToSpaceNDOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def BatchToSpaceNDOptionsStart(builder):
    builder.StartObject(0)


def BatchToSpaceNDOptionsEnd(builder):
    return builder.EndObject()


class BatchToSpaceNDOptionsT(object):

    # BatchToSpaceNDOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        batchToSpaceNDOptions = BatchToSpaceNDOptions()
        batchToSpaceNDOptions.Init(buf, pos)
        return cls.InitFromObj(batchToSpaceNDOptions)

    @classmethod
    def InitFromObj(cls, batchToSpaceNDOptions):
        x = BatchToSpaceNDOptionsT()
        x._UnPack(batchToSpaceNDOptions)
        return x

    # BatchToSpaceNDOptionsT
    def _UnPack(self, batchToSpaceNDOptions):
        if batchToSpaceNDOptions is None:
            return

    # BatchToSpaceNDOptionsT
    def Pack(self, builder):
        BatchToSpaceNDOptionsStart(builder)
        batchToSpaceNDOptions = BatchToSpaceNDOptionsEnd(builder)
        return batchToSpaceNDOptions


class NonMaxSuppressionV4Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsNonMaxSuppressionV4Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NonMaxSuppressionV4Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NonMaxSuppressionV4OptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # NonMaxSuppressionV4Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NonMaxSuppressionV4OptionsStart(builder):
    builder.StartObject(0)


def NonMaxSuppressionV4OptionsEnd(builder):
    return builder.EndObject()


class NonMaxSuppressionV4OptionsT(object):

    # NonMaxSuppressionV4OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        nonMaxSuppressionV4Options = NonMaxSuppressionV4Options()
        nonMaxSuppressionV4Options.Init(buf, pos)
        return cls.InitFromObj(nonMaxSuppressionV4Options)

    @classmethod
    def InitFromObj(cls, nonMaxSuppressionV4Options):
        x = NonMaxSuppressionV4OptionsT()
        x._UnPack(nonMaxSuppressionV4Options)
        return x

    # NonMaxSuppressionV4OptionsT
    def _UnPack(self, nonMaxSuppressionV4Options):
        if nonMaxSuppressionV4Options is None:
            return

    # NonMaxSuppressionV4OptionsT
    def Pack(self, builder):
        NonMaxSuppressionV4OptionsStart(builder)
        nonMaxSuppressionV4Options = NonMaxSuppressionV4OptionsEnd(builder)
        return nonMaxSuppressionV4Options


class CallOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsCallOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CallOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CallOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # CallOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CallOptions
    def Subgraph(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Uint32Flags, o + self._tab.Pos
            )
        return 0


def CallOptionsStart(builder):
    builder.StartObject(1)


def CallOptionsAddSubgraph(builder, subgraph):
    builder.PrependUint32Slot(0, subgraph, 0)


def CallOptionsEnd(builder):
    return builder.EndObject()


class CallOptionsT(object):

    # CallOptionsT
    def __init__(self):
        self.subgraph = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        callOptions = CallOptions()
        callOptions.Init(buf, pos)
        return cls.InitFromObj(callOptions)

    @classmethod
    def InitFromObj(cls, callOptions):
        x = CallOptionsT()
        x._UnPack(callOptions)
        return x

    # CallOptionsT
    def _UnPack(self, callOptions):
        if callOptions is None:
            return
        self.subgraph = callOptions.Subgraph()

    # CallOptionsT
    def Pack(self, builder):
        CallOptionsStart(builder)
        CallOptionsAddSubgraph(builder, self.subgraph)
        callOptions = CallOptionsEnd(builder)
        return callOptions


class GatherOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsGatherOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GatherOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # GatherOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GatherOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def GatherOptionsStart(builder):
    builder.StartObject(1)


def GatherOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)


def GatherOptionsEnd(builder):
    return builder.EndObject()


class GatherOptionsT(object):

    # GatherOptionsT
    def __init__(self):
        self.axis = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        gatherOptions = GatherOptions()
        gatherOptions.Init(buf, pos)
        return cls.InitFromObj(gatherOptions)

    @classmethod
    def InitFromObj(cls, gatherOptions):
        x = GatherOptionsT()
        x._UnPack(gatherOptions)
        return x

    # GatherOptionsT
    def _UnPack(self, gatherOptions):
        if gatherOptions is None:
            return
        self.axis = gatherOptions.Axis()

    # GatherOptionsT
    def Pack(self, builder):
        GatherOptionsStart(builder)
        GatherOptionsAddAxis(builder, self.axis)
        gatherOptions = GatherOptionsEnd(builder)
        return gatherOptions


class MirrorPadOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsMirrorPadOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MirrorPadOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MirrorPadOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # MirrorPadOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MirrorPadOptions
    def Mode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def MirrorPadOptionsStart(builder):
    builder.StartObject(1)


def MirrorPadOptionsAddMode(builder, mode):
    builder.PrependInt8Slot(0, mode, 0)


def MirrorPadOptionsEnd(builder):
    return builder.EndObject()


class MirrorPadOptionsT(object):

    # MirrorPadOptionsT
    def __init__(self):
        self.mode = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        mirrorPadOptions = MirrorPadOptions()
        mirrorPadOptions.Init(buf, pos)
        return cls.InitFromObj(mirrorPadOptions)

    @classmethod
    def InitFromObj(cls, mirrorPadOptions):
        x = MirrorPadOptionsT()
        x._UnPack(mirrorPadOptions)
        return x

    # MirrorPadOptionsT
    def _UnPack(self, mirrorPadOptions):
        if mirrorPadOptions is None:
            return
        self.mode = mirrorPadOptions.Mode()

    # MirrorPadOptionsT
    def Pack(self, builder):
        MirrorPadOptionsStart(builder)
        MirrorPadOptionsAddMode(builder, self.mode)
        mirrorPadOptions = MirrorPadOptionsEnd(builder)
        return mirrorPadOptions


class WhileOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsWhileOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhileOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def WhileOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # WhileOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # WhileOptions
    def CondSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # WhileOptions
    def BodySubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def WhileOptionsStart(builder):
    builder.StartObject(2)


def WhileOptionsAddCondSubgraphIndex(builder, condSubgraphIndex):
    builder.PrependInt32Slot(0, condSubgraphIndex, 0)


def WhileOptionsAddBodySubgraphIndex(builder, bodySubgraphIndex):
    builder.PrependInt32Slot(1, bodySubgraphIndex, 0)


def WhileOptionsEnd(builder):
    return builder.EndObject()


class WhileOptionsT(object):

    # WhileOptionsT
    def __init__(self):
        self.condSubgraphIndex = 0  # type: int
        self.bodySubgraphIndex = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        whileOptions = WhileOptions()
        whileOptions.Init(buf, pos)
        return cls.InitFromObj(whileOptions)

    @classmethod
    def InitFromObj(cls, whileOptions):
        x = WhileOptionsT()
        x._UnPack(whileOptions)
        return x

    # WhileOptionsT
    def _UnPack(self, whileOptions):
        if whileOptions is None:
            return
        self.condSubgraphIndex = whileOptions.CondSubgraphIndex()
        self.bodySubgraphIndex = whileOptions.BodySubgraphIndex()

    # WhileOptionsT
    def Pack(self, builder):
        WhileOptionsStart(builder)
        WhileOptionsAddCondSubgraphIndex(builder, self.condSubgraphIndex)
        WhileOptionsAddBodySubgraphIndex(builder, self.bodySubgraphIndex)
        whileOptions = WhileOptionsEnd(builder)
        return whileOptions


class StridedSliceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsStridedSliceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = StridedSliceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def StridedSliceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # StridedSliceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # StridedSliceOptions
    def BeginMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def EndMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def EllipsisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def NewAxisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def ShrinkAxisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def StridedSliceOptionsStart(builder):
    builder.StartObject(5)


def StridedSliceOptionsAddBeginMask(builder, beginMask):
    builder.PrependInt32Slot(0, beginMask, 0)


def StridedSliceOptionsAddEndMask(builder, endMask):
    builder.PrependInt32Slot(1, endMask, 0)


def StridedSliceOptionsAddEllipsisMask(builder, ellipsisMask):
    builder.PrependInt32Slot(2, ellipsisMask, 0)


def StridedSliceOptionsAddNewAxisMask(builder, newAxisMask):
    builder.PrependInt32Slot(3, newAxisMask, 0)


def StridedSliceOptionsAddShrinkAxisMask(builder, shrinkAxisMask):
    builder.PrependInt32Slot(4, shrinkAxisMask, 0)


def StridedSliceOptionsEnd(builder):
    return builder.EndObject()


class StridedSliceOptionsT(object):

    # StridedSliceOptionsT
    def __init__(self):
        self.beginMask = 0  # type: int
        self.endMask = 0  # type: int
        self.ellipsisMask = 0  # type: int
        self.newAxisMask = 0  # type: int
        self.shrinkAxisMask = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        stridedSliceOptions = StridedSliceOptions()
        stridedSliceOptions.Init(buf, pos)
        return cls.InitFromObj(stridedSliceOptions)

    @classmethod
    def InitFromObj(cls, stridedSliceOptions):
        x = StridedSliceOptionsT()
        x._UnPack(stridedSliceOptions)
        return x

    # StridedSliceOptionsT
    def _UnPack(self, stridedSliceOptions):
        if stridedSliceOptions is None:
            return
        self.beginMask = stridedSliceOptions.BeginMask()
        self.endMask = stridedSliceOptions.EndMask()
        self.ellipsisMask = stridedSliceOptions.EllipsisMask()
        self.newAxisMask = stridedSliceOptions.NewAxisMask()
        self.shrinkAxisMask = stridedSliceOptions.ShrinkAxisMask()

    # StridedSliceOptionsT
    def Pack(self, builder):
        StridedSliceOptionsStart(builder)
        StridedSliceOptionsAddBeginMask(builder, self.beginMask)
        StridedSliceOptionsAddEndMask(builder, self.endMask)
        StridedSliceOptionsAddEllipsisMask(builder, self.ellipsisMask)
        StridedSliceOptionsAddNewAxisMask(builder, self.newAxisMask)
        StridedSliceOptionsAddShrinkAxisMask(builder, self.shrinkAxisMask)
        stridedSliceOptions = StridedSliceOptionsEnd(builder)
        return stridedSliceOptions


class ConcatEmbeddingsOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsConcatEmbeddingsOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConcatEmbeddingsOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ConcatEmbeddingsOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ConcatEmbeddingsOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConcatEmbeddingsOptions
    def NumChannels(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannel(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannel(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0


def ConcatEmbeddingsOptionsStart(builder):
    builder.StartObject(3)


def ConcatEmbeddingsOptionsAddNumChannels(builder, numChannels):
    builder.PrependInt32Slot(0, numChannels, 0)


def ConcatEmbeddingsOptionsAddNumColumnsPerChannel(builder, numColumnsPerChannel):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(numColumnsPerChannel), 0
    )


def ConcatEmbeddingsOptionsStartNumColumnsPerChannelVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(builder, embeddingDimPerChannel):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(embeddingDimPerChannel), 0
    )


def ConcatEmbeddingsOptionsStartEmbeddingDimPerChannelVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def ConcatEmbeddingsOptionsEnd(builder):
    return builder.EndObject()


class ConcatEmbeddingsOptionsT(object):

    # ConcatEmbeddingsOptionsT
    def __init__(self):
        self.numChannels = 0  # type: int
        self.numColumnsPerChannel = None  # type: List[int]
        self.embeddingDimPerChannel = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        concatEmbeddingsOptions = ConcatEmbeddingsOptions()
        concatEmbeddingsOptions.Init(buf, pos)
        return cls.InitFromObj(concatEmbeddingsOptions)

    @classmethod
    def InitFromObj(cls, concatEmbeddingsOptions):
        x = ConcatEmbeddingsOptionsT()
        x._UnPack(concatEmbeddingsOptions)
        return x

    # ConcatEmbeddingsOptionsT
    def _UnPack(self, concatEmbeddingsOptions):
        if concatEmbeddingsOptions is None:
            return
        self.numChannels = concatEmbeddingsOptions.NumChannels()
        if not concatEmbeddingsOptions.NumColumnsPerChannelIsNone():
            if np is None:
                self.numColumnsPerChannel = []
                for i in range(concatEmbeddingsOptions.NumColumnsPerChannelLength()):
                    self.numColumnsPerChannel.append(
                        concatEmbeddingsOptions.NumColumnsPerChannel(i)
                    )
            else:
                self.numColumnsPerChannel = (
                    concatEmbeddingsOptions.NumColumnsPerChannelAsNumpy()
                )
        if not concatEmbeddingsOptions.EmbeddingDimPerChannelIsNone():
            if np is None:
                self.embeddingDimPerChannel = []
                for i in range(concatEmbeddingsOptions.EmbeddingDimPerChannelLength()):
                    self.embeddingDimPerChannel.append(
                        concatEmbeddingsOptions.EmbeddingDimPerChannel(i)
                    )
            else:
                self.embeddingDimPerChannel = (
                    concatEmbeddingsOptions.EmbeddingDimPerChannelAsNumpy()
                )

    # ConcatEmbeddingsOptionsT
    def Pack(self, builder):
        if self.numColumnsPerChannel is not None:
            if np is not None and type(self.numColumnsPerChannel) is np.ndarray:
                numColumnsPerChannel = builder.CreateNumpyVector(
                    self.numColumnsPerChannel
                )
            else:
                ConcatEmbeddingsOptionsStartNumColumnsPerChannelVector(
                    builder, len(self.numColumnsPerChannel)
                )
                for i in reversed(range(len(self.numColumnsPerChannel))):
                    builder.PrependInt32(self.numColumnsPerChannel[i])
                numColumnsPerChannel = builder.EndVector(len(self.numColumnsPerChannel))
        if self.embeddingDimPerChannel is not None:
            if np is not None and type(self.embeddingDimPerChannel) is np.ndarray:
                embeddingDimPerChannel = builder.CreateNumpyVector(
                    self.embeddingDimPerChannel
                )
            else:
                ConcatEmbeddingsOptionsStartEmbeddingDimPerChannelVector(
                    builder, len(self.embeddingDimPerChannel)
                )
                for i in reversed(range(len(self.embeddingDimPerChannel))):
                    builder.PrependInt32(self.embeddingDimPerChannel[i])
                embeddingDimPerChannel = builder.EndVector(
                    len(self.embeddingDimPerChannel)
                )
        ConcatEmbeddingsOptionsStart(builder)
        ConcatEmbeddingsOptionsAddNumChannels(builder, self.numChannels)
        if self.numColumnsPerChannel is not None:
            ConcatEmbeddingsOptionsAddNumColumnsPerChannel(
                builder, numColumnsPerChannel
            )
        if self.embeddingDimPerChannel is not None:
            ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(
                builder, embeddingDimPerChannel
            )
        concatEmbeddingsOptions = ConcatEmbeddingsOptionsEnd(builder)
        return concatEmbeddingsOptions


class IfOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsIfOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = IfOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def IfOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # IfOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # IfOptions
    def ThenSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # IfOptions
    def ElseSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def IfOptionsStart(builder):
    builder.StartObject(2)


def IfOptionsAddThenSubgraphIndex(builder, thenSubgraphIndex):
    builder.PrependInt32Slot(0, thenSubgraphIndex, 0)


def IfOptionsAddElseSubgraphIndex(builder, elseSubgraphIndex):
    builder.PrependInt32Slot(1, elseSubgraphIndex, 0)


def IfOptionsEnd(builder):
    return builder.EndObject()


class IfOptionsT(object):

    # IfOptionsT
    def __init__(self):
        self.thenSubgraphIndex = 0  # type: int
        self.elseSubgraphIndex = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        ifOptions = IfOptions()
        ifOptions.Init(buf, pos)
        return cls.InitFromObj(ifOptions)

    @classmethod
    def InitFromObj(cls, ifOptions):
        x = IfOptionsT()
        x._UnPack(ifOptions)
        return x

    # IfOptionsT
    def _UnPack(self, ifOptions):
        if ifOptions is None:
            return
        self.thenSubgraphIndex = ifOptions.ThenSubgraphIndex()
        self.elseSubgraphIndex = ifOptions.ElseSubgraphIndex()

    # IfOptionsT
    def Pack(self, builder):
        IfOptionsStart(builder)
        IfOptionsAddThenSubgraphIndex(builder, self.thenSubgraphIndex)
        IfOptionsAddElseSubgraphIndex(builder, self.elseSubgraphIndex)
        ifOptions = IfOptionsEnd(builder)
        return ifOptions


class PadOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsPadOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PadOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def PadOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # PadOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def PadOptionsStart(builder):
    builder.StartObject(0)


def PadOptionsEnd(builder):
    return builder.EndObject()


class PadOptionsT(object):

    # PadOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        padOptions = PadOptions()
        padOptions.Init(buf, pos)
        return cls.InitFromObj(padOptions)

    @classmethod
    def InitFromObj(cls, padOptions):
        x = PadOptionsT()
        x._UnPack(padOptions)
        return x

    # PadOptionsT
    def _UnPack(self, padOptions):
        if padOptions is None:
            return

    # PadOptionsT
    def Pack(self, builder):
        PadOptionsStart(builder)
        padOptions = PadOptionsEnd(builder)
        return padOptions


class DepthwiseConv2DOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDepthwiseConv2DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthwiseConv2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DepthwiseConv2DOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DepthwiseConv2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthwiseConv2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DepthMultiplier(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # DepthwiseConv2DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1


def DepthwiseConv2DOptionsStart(builder):
    builder.StartObject(7)


def DepthwiseConv2DOptionsAddPadding(builder, padding):
    builder.PrependInt8Slot(0, padding, 0)


def DepthwiseConv2DOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(1, strideW, 0)


def DepthwiseConv2DOptionsAddStrideH(builder, strideH):
    builder.PrependInt32Slot(2, strideH, 0)


def DepthwiseConv2DOptionsAddDepthMultiplier(builder, depthMultiplier):
    builder.PrependInt32Slot(3, depthMultiplier, 0)


def DepthwiseConv2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(4, fusedActivationFunction, 0)


def DepthwiseConv2DOptionsAddDilationWFactor(builder, dilationWFactor):
    builder.PrependInt32Slot(5, dilationWFactor, 1)


def DepthwiseConv2DOptionsAddDilationHFactor(builder, dilationHFactor):
    builder.PrependInt32Slot(6, dilationHFactor, 1)


def DepthwiseConv2DOptionsEnd(builder):
    return builder.EndObject()


class DepthwiseConv2DOptionsT(object):

    # DepthwiseConv2DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.depthMultiplier = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.dilationWFactor = 1  # type: int
        self.dilationHFactor = 1  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        depthwiseConv2DOptions = DepthwiseConv2DOptions()
        depthwiseConv2DOptions.Init(buf, pos)
        return cls.InitFromObj(depthwiseConv2DOptions)

    @classmethod
    def InitFromObj(cls, depthwiseConv2DOptions):
        x = DepthwiseConv2DOptionsT()
        x._UnPack(depthwiseConv2DOptions)
        return x

    # DepthwiseConv2DOptionsT
    def _UnPack(self, depthwiseConv2DOptions):
        if depthwiseConv2DOptions is None:
            return
        self.padding = depthwiseConv2DOptions.Padding()
        self.strideW = depthwiseConv2DOptions.StrideW()
        self.strideH = depthwiseConv2DOptions.StrideH()
        self.depthMultiplier = depthwiseConv2DOptions.DepthMultiplier()
        self.fusedActivationFunction = depthwiseConv2DOptions.FusedActivationFunction()
        self.dilationWFactor = depthwiseConv2DOptions.DilationWFactor()
        self.dilationHFactor = depthwiseConv2DOptions.DilationHFactor()

    # DepthwiseConv2DOptionsT
    def Pack(self, builder):
        DepthwiseConv2DOptionsStart(builder)
        DepthwiseConv2DOptionsAddPadding(builder, self.padding)
        DepthwiseConv2DOptionsAddStrideW(builder, self.strideW)
        DepthwiseConv2DOptionsAddStrideH(builder, self.strideH)
        DepthwiseConv2DOptionsAddDepthMultiplier(builder, self.depthMultiplier)
        DepthwiseConv2DOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        DepthwiseConv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        DepthwiseConv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        depthwiseConv2DOptions = DepthwiseConv2DOptionsEnd(builder)
        return depthwiseConv2DOptions


class SubGraph(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSubGraph(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubGraph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SubGraphBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SubGraph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubGraph
    def Tensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def TensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def TensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SubGraph
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SubGraph
    def InputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SubGraph
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SubGraph
    def OutputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # SubGraph
    def Operators(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Operator()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def OperatorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OperatorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # SubGraph
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None


def SubGraphStart(builder):
    builder.StartObject(5)


def SubGraphAddTensors(builder, tensors):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0
    )


def SubGraphStartTensorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SubGraphAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0
    )


def SubGraphStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SubGraphAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0
    )


def SubGraphStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SubGraphAddOperators(builder, operators):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(operators), 0
    )


def SubGraphStartOperatorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def SubGraphAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0
    )


def SubGraphEnd(builder):
    return builder.EndObject()


class SubGraphT(object):

    # SubGraphT
    def __init__(self):
        self.tensors = None  # type: List[TensorT]
        self.inputs = None  # type: List[int]
        self.outputs = None  # type: List[int]
        self.operators = None  # type: List[OperatorT]
        self.name = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        subGraph = SubGraph()
        subGraph.Init(buf, pos)
        return cls.InitFromObj(subGraph)

    @classmethod
    def InitFromObj(cls, subGraph):
        x = SubGraphT()
        x._UnPack(subGraph)
        return x

    # SubGraphT
    def _UnPack(self, subGraph):
        if subGraph is None:
            return
        if not subGraph.TensorsIsNone():
            self.tensors = []
            for i in range(subGraph.TensorsLength()):
                if subGraph.Tensors(i) is None:
                    self.tensors.append(None)
                else:
                    tensor_ = TensorT.InitFromObj(subGraph.Tensors(i))
                    self.tensors.append(tensor_)
        if not subGraph.InputsIsNone():
            if np is None:
                self.inputs = []
                for i in range(subGraph.InputsLength()):
                    self.inputs.append(subGraph.Inputs(i))
            else:
                self.inputs = subGraph.InputsAsNumpy()
        if not subGraph.OutputsIsNone():
            if np is None:
                self.outputs = []
                for i in range(subGraph.OutputsLength()):
                    self.outputs.append(subGraph.Outputs(i))
            else:
                self.outputs = subGraph.OutputsAsNumpy()
        if not subGraph.OperatorsIsNone():
            self.operators = []
            for i in range(subGraph.OperatorsLength()):
                if subGraph.Operators(i) is None:
                    self.operators.append(None)
                else:
                    operator_ = OperatorT.InitFromObj(subGraph.Operators(i))
                    self.operators.append(operator_)
        self.name = subGraph.Name()

    # SubGraphT
    def Pack(self, builder):
        if self.tensors is not None:
            tensorslist = []
            for i in range(len(self.tensors)):
                tensorslist.append(self.tensors[i].Pack(builder))
            SubGraphStartTensorsVector(builder, len(self.tensors))
            for i in reversed(range(len(self.tensors))):
                builder.PrependUOffsetTRelative(tensorslist[i])
            tensors = builder.EndVector(len(self.tensors))
        if self.inputs is not None:
            if np is not None and type(self.inputs) is np.ndarray:
                inputs = builder.CreateNumpyVector(self.inputs)
            else:
                SubGraphStartInputsVector(builder, len(self.inputs))
                for i in reversed(range(len(self.inputs))):
                    builder.PrependInt32(self.inputs[i])
                inputs = builder.EndVector(len(self.inputs))
        if self.outputs is not None:
            if np is not None and type(self.outputs) is np.ndarray:
                outputs = builder.CreateNumpyVector(self.outputs)
            else:
                SubGraphStartOutputsVector(builder, len(self.outputs))
                for i in reversed(range(len(self.outputs))):
                    builder.PrependInt32(self.outputs[i])
                outputs = builder.EndVector(len(self.outputs))
        if self.operators is not None:
            operatorslist = []
            for i in range(len(self.operators)):
                operatorslist.append(self.operators[i].Pack(builder))
            SubGraphStartOperatorsVector(builder, len(self.operators))
            for i in reversed(range(len(self.operators))):
                builder.PrependUOffsetTRelative(operatorslist[i])
            operators = builder.EndVector(len(self.operators))
        if self.name is not None:
            name = builder.CreateString(self.name)
        SubGraphStart(builder)
        if self.tensors is not None:
            SubGraphAddTensors(builder, tensors)
        if self.inputs is not None:
            SubGraphAddInputs(builder, inputs)
        if self.outputs is not None:
            SubGraphAddOutputs(builder, outputs)
        if self.operators is not None:
            SubGraphAddOperators(builder, operators)
        if self.name is not None:
            SubGraphAddName(builder, name)
        subGraph = SubGraphEnd(builder)
        return subGraph


class UnpackOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsUnpackOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UnpackOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def UnpackOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # UnpackOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UnpackOptions
    def Num(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UnpackOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def UnpackOptionsStart(builder):
    builder.StartObject(2)


def UnpackOptionsAddNum(builder, num):
    builder.PrependInt32Slot(0, num, 0)


def UnpackOptionsAddAxis(builder, axis):
    builder.PrependInt32Slot(1, axis, 0)


def UnpackOptionsEnd(builder):
    return builder.EndObject()


class UnpackOptionsT(object):

    # UnpackOptionsT
    def __init__(self):
        self.num = 0  # type: int
        self.axis = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        unpackOptions = UnpackOptions()
        unpackOptions.Init(buf, pos)
        return cls.InitFromObj(unpackOptions)

    @classmethod
    def InitFromObj(cls, unpackOptions):
        x = UnpackOptionsT()
        x._UnPack(unpackOptions)
        return x

    # UnpackOptionsT
    def _UnPack(self, unpackOptions):
        if unpackOptions is None:
            return
        self.num = unpackOptions.Num()
        self.axis = unpackOptions.Axis()

    # UnpackOptionsT
    def Pack(self, builder):
        UnpackOptionsStart(builder)
        UnpackOptionsAddNum(builder, self.num)
        UnpackOptionsAddAxis(builder, self.axis)
        unpackOptions = UnpackOptionsEnd(builder)
        return unpackOptions


class BatchMatMulOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsBatchMatMulOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BatchMatMulOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BatchMatMulOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # BatchMatMulOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BatchMatMulOptions
    def AdjX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # BatchMatMulOptions
    def AdjY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def BatchMatMulOptionsStart(builder):
    builder.StartObject(2)


def BatchMatMulOptionsAddAdjX(builder, adjX):
    builder.PrependBoolSlot(0, adjX, 0)


def BatchMatMulOptionsAddAdjY(builder, adjY):
    builder.PrependBoolSlot(1, adjY, 0)


def BatchMatMulOptionsEnd(builder):
    return builder.EndObject()


class BatchMatMulOptionsT(object):

    # BatchMatMulOptionsT
    def __init__(self):
        self.adjX = False  # type: bool
        self.adjY = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        batchMatMulOptions = BatchMatMulOptions()
        batchMatMulOptions.Init(buf, pos)
        return cls.InitFromObj(batchMatMulOptions)

    @classmethod
    def InitFromObj(cls, batchMatMulOptions):
        x = BatchMatMulOptionsT()
        x._UnPack(batchMatMulOptions)
        return x

    # BatchMatMulOptionsT
    def _UnPack(self, batchMatMulOptions):
        if batchMatMulOptions is None:
            return
        self.adjX = batchMatMulOptions.AdjX()
        self.adjY = batchMatMulOptions.AdjY()

    # BatchMatMulOptionsT
    def Pack(self, builder):
        BatchMatMulOptionsStart(builder)
        BatchMatMulOptionsAddAdjX(builder, self.adjX)
        BatchMatMulOptionsAddAdjY(builder, self.adjY)
        batchMatMulOptions = BatchMatMulOptionsEnd(builder)
        return batchMatMulOptions


class SubOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSubOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SubOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SubOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # SubOptions
    def PotScaleInt16(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return True


def SubOptionsStart(builder):
    builder.StartObject(2)


def SubOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def SubOptionsAddPotScaleInt16(builder, potScaleInt16):
    builder.PrependBoolSlot(1, potScaleInt16, 1)


def SubOptionsEnd(builder):
    return builder.EndObject()


class SubOptionsT(object):

    # SubOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.potScaleInt16 = True  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        subOptions = SubOptions()
        subOptions.Init(buf, pos)
        return cls.InitFromObj(subOptions)

    @classmethod
    def InitFromObj(cls, subOptions):
        x = SubOptionsT()
        x._UnPack(subOptions)
        return x

    # SubOptionsT
    def _UnPack(self, subOptions):
        if subOptions is None:
            return
        self.fusedActivationFunction = subOptions.FusedActivationFunction()
        self.potScaleInt16 = subOptions.PotScaleInt16()

    # SubOptionsT
    def Pack(self, builder):
        SubOptionsStart(builder)
        SubOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SubOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        subOptions = SubOptionsEnd(builder)
        return subOptions


class TransposeConvOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsTransposeConvOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TransposeConvOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def TransposeConvOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # TransposeConvOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TransposeConvOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # TransposeConvOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TransposeConvOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def TransposeConvOptionsStart(builder):
    builder.StartObject(3)


def TransposeConvOptionsAddPadding(builder, padding):
    builder.PrependInt8Slot(0, padding, 0)


def TransposeConvOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(1, strideW, 0)


def TransposeConvOptionsAddStrideH(builder, strideH):
    builder.PrependInt32Slot(2, strideH, 0)


def TransposeConvOptionsEnd(builder):
    return builder.EndObject()


class TransposeConvOptionsT(object):

    # TransposeConvOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        transposeConvOptions = TransposeConvOptions()
        transposeConvOptions.Init(buf, pos)
        return cls.InitFromObj(transposeConvOptions)

    @classmethod
    def InitFromObj(cls, transposeConvOptions):
        x = TransposeConvOptionsT()
        x._UnPack(transposeConvOptions)
        return x

    # TransposeConvOptionsT
    def _UnPack(self, transposeConvOptions):
        if transposeConvOptions is None:
            return
        self.padding = transposeConvOptions.Padding()
        self.strideW = transposeConvOptions.StrideW()
        self.strideH = transposeConvOptions.StrideH()

    # TransposeConvOptionsT
    def Pack(self, builder):
        TransposeConvOptionsStart(builder)
        TransposeConvOptionsAddPadding(builder, self.padding)
        TransposeConvOptionsAddStrideW(builder, self.strideW)
        TransposeConvOptionsAddStrideH(builder, self.strideH)
        transposeConvOptions = TransposeConvOptionsEnd(builder)
        return transposeConvOptions


class DepthToSpaceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDepthToSpaceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthToSpaceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DepthToSpaceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DepthToSpaceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthToSpaceOptions
    def BlockSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def DepthToSpaceOptionsStart(builder):
    builder.StartObject(1)


def DepthToSpaceOptionsAddBlockSize(builder, blockSize):
    builder.PrependInt32Slot(0, blockSize, 0)


def DepthToSpaceOptionsEnd(builder):
    return builder.EndObject()


class DepthToSpaceOptionsT(object):

    # DepthToSpaceOptionsT
    def __init__(self):
        self.blockSize = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        depthToSpaceOptions = DepthToSpaceOptions()
        depthToSpaceOptions.Init(buf, pos)
        return cls.InitFromObj(depthToSpaceOptions)

    @classmethod
    def InitFromObj(cls, depthToSpaceOptions):
        x = DepthToSpaceOptionsT()
        x._UnPack(depthToSpaceOptions)
        return x

    # DepthToSpaceOptionsT
    def _UnPack(self, depthToSpaceOptions):
        if depthToSpaceOptions is None:
            return
        self.blockSize = depthToSpaceOptions.BlockSize()

    # DepthToSpaceOptionsT
    def Pack(self, builder):
        DepthToSpaceOptionsStart(builder)
        DepthToSpaceOptionsAddBlockSize(builder, self.blockSize)
        depthToSpaceOptions = DepthToSpaceOptionsEnd(builder)
        return depthToSpaceOptions


class Padding(object):
    SAME = 0
    VALID = 1


class Conv2DOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsConv2DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Conv2DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Conv2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv2DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1


def Conv2DOptionsStart(builder):
    builder.StartObject(6)


def Conv2DOptionsAddPadding(builder, padding):
    builder.PrependInt8Slot(0, padding, 0)


def Conv2DOptionsAddStrideW(builder, strideW):
    builder.PrependInt32Slot(1, strideW, 0)


def Conv2DOptionsAddStrideH(builder, strideH):
    builder.PrependInt32Slot(2, strideH, 0)


def Conv2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(3, fusedActivationFunction, 0)


def Conv2DOptionsAddDilationWFactor(builder, dilationWFactor):
    builder.PrependInt32Slot(4, dilationWFactor, 1)


def Conv2DOptionsAddDilationHFactor(builder, dilationHFactor):
    builder.PrependInt32Slot(5, dilationHFactor, 1)


def Conv2DOptionsEnd(builder):
    return builder.EndObject()


class Conv2DOptionsT(object):

    # Conv2DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.dilationWFactor = 1  # type: int
        self.dilationHFactor = 1  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv2DOptions = Conv2DOptions()
        conv2DOptions.Init(buf, pos)
        return cls.InitFromObj(conv2DOptions)

    @classmethod
    def InitFromObj(cls, conv2DOptions):
        x = Conv2DOptionsT()
        x._UnPack(conv2DOptions)
        return x

    # Conv2DOptionsT
    def _UnPack(self, conv2DOptions):
        if conv2DOptions is None:
            return
        self.padding = conv2DOptions.Padding()
        self.strideW = conv2DOptions.StrideW()
        self.strideH = conv2DOptions.StrideH()
        self.fusedActivationFunction = conv2DOptions.FusedActivationFunction()
        self.dilationWFactor = conv2DOptions.DilationWFactor()
        self.dilationHFactor = conv2DOptions.DilationHFactor()

    # Conv2DOptionsT
    def Pack(self, builder):
        Conv2DOptionsStart(builder)
        Conv2DOptionsAddPadding(builder, self.padding)
        Conv2DOptionsAddStrideW(builder, self.strideW)
        Conv2DOptionsAddStrideH(builder, self.strideH)
        Conv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        Conv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        Conv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        conv2DOptions = Conv2DOptionsEnd(builder)
        return conv2DOptions


class SparseIndexVector(object):
    NONE = 0
    Int32Vector = 1
    Uint16Vector = 2
    Uint8Vector = 3


def SparseIndexVectorCreator(unionType, table):
    from flatbuffers.table import Table

    if not isinstance(table, Table):
        return None
    if unionType == SparseIndexVector().Int32Vector:
        return Int32VectorT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == SparseIndexVector().Uint16Vector:
        return Uint16VectorT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == SparseIndexVector().Uint8Vector:
        return Uint8VectorT.InitFromBuf(table.Bytes, table.Pos)
    return None


class DequantizeOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDequantizeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DequantizeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DequantizeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # DequantizeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def DequantizeOptionsStart(builder):
    builder.StartObject(0)


def DequantizeOptionsEnd(builder):
    return builder.EndObject()


class DequantizeOptionsT(object):

    # DequantizeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dequantizeOptions = DequantizeOptions()
        dequantizeOptions.Init(buf, pos)
        return cls.InitFromObj(dequantizeOptions)

    @classmethod
    def InitFromObj(cls, dequantizeOptions):
        x = DequantizeOptionsT()
        x._UnPack(dequantizeOptions)
        return x

    # DequantizeOptionsT
    def _UnPack(self, dequantizeOptions):
        if dequantizeOptions is None:
            return

    # DequantizeOptionsT
    def Pack(self, builder):
        DequantizeOptionsStart(builder)
        dequantizeOptions = DequantizeOptionsEnd(builder)
        return dequantizeOptions


class WhereOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsWhereOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhereOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def WhereOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # WhereOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def WhereOptionsStart(builder):
    builder.StartObject(0)


def WhereOptionsEnd(builder):
    return builder.EndObject()


class WhereOptionsT(object):

    # WhereOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        whereOptions = WhereOptions()
        whereOptions.Init(buf, pos)
        return cls.InitFromObj(whereOptions)

    @classmethod
    def InitFromObj(cls, whereOptions):
        x = WhereOptionsT()
        x._UnPack(whereOptions)
        return x

    # WhereOptionsT
    def _UnPack(self, whereOptions):
        if whereOptions is None:
            return

    # WhereOptionsT
    def Pack(self, builder):
        WhereOptionsStart(builder)
        whereOptions = WhereOptionsEnd(builder)
        return whereOptions


class ExpandDimsOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsExpandDimsOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ExpandDimsOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ExpandDimsOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ExpandDimsOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ExpandDimsOptionsStart(builder):
    builder.StartObject(0)


def ExpandDimsOptionsEnd(builder):
    return builder.EndObject()


class ExpandDimsOptionsT(object):

    # ExpandDimsOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        expandDimsOptions = ExpandDimsOptions()
        expandDimsOptions.Init(buf, pos)
        return cls.InitFromObj(expandDimsOptions)

    @classmethod
    def InitFromObj(cls, expandDimsOptions):
        x = ExpandDimsOptionsT()
        x._UnPack(expandDimsOptions)
        return x

    # ExpandDimsOptionsT
    def _UnPack(self, expandDimsOptions):
        if expandDimsOptions is None:
            return

    # ExpandDimsOptionsT
    def Pack(self, builder):
        ExpandDimsOptionsStart(builder)
        expandDimsOptions = ExpandDimsOptionsEnd(builder)
        return expandDimsOptions


class SpaceToDepthOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSpaceToDepthOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SpaceToDepthOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SpaceToDepthOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SpaceToDepthOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SpaceToDepthOptions
    def BlockSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def SpaceToDepthOptionsStart(builder):
    builder.StartObject(1)


def SpaceToDepthOptionsAddBlockSize(builder, blockSize):
    builder.PrependInt32Slot(0, blockSize, 0)


def SpaceToDepthOptionsEnd(builder):
    return builder.EndObject()


class SpaceToDepthOptionsT(object):

    # SpaceToDepthOptionsT
    def __init__(self):
        self.blockSize = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        spaceToDepthOptions = SpaceToDepthOptions()
        spaceToDepthOptions.Init(buf, pos)
        return cls.InitFromObj(spaceToDepthOptions)

    @classmethod
    def InitFromObj(cls, spaceToDepthOptions):
        x = SpaceToDepthOptionsT()
        x._UnPack(spaceToDepthOptions)
        return x

    # SpaceToDepthOptionsT
    def _UnPack(self, spaceToDepthOptions):
        if spaceToDepthOptions is None:
            return
        self.blockSize = spaceToDepthOptions.BlockSize()

    # SpaceToDepthOptionsT
    def Pack(self, builder):
        SpaceToDepthOptionsStart(builder)
        SpaceToDepthOptionsAddBlockSize(builder, self.blockSize)
        spaceToDepthOptions = SpaceToDepthOptionsEnd(builder)
        return spaceToDepthOptions


class L2NormOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsL2NormOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = L2NormOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def L2NormOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # L2NormOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # L2NormOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def L2NormOptionsStart(builder):
    builder.StartObject(1)


def L2NormOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def L2NormOptionsEnd(builder):
    return builder.EndObject()


class L2NormOptionsT(object):

    # L2NormOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        l2NormOptions = L2NormOptions()
        l2NormOptions.Init(buf, pos)
        return cls.InitFromObj(l2NormOptions)

    @classmethod
    def InitFromObj(cls, l2NormOptions):
        x = L2NormOptionsT()
        x._UnPack(l2NormOptions)
        return x

    # L2NormOptionsT
    def _UnPack(self, l2NormOptions):
        if l2NormOptions is None:
            return
        self.fusedActivationFunction = l2NormOptions.FusedActivationFunction()

    # L2NormOptionsT
    def Pack(self, builder):
        L2NormOptionsStart(builder)
        L2NormOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        l2NormOptions = L2NormOptionsEnd(builder)
        return l2NormOptions


class LSTMKernelType(object):
    FULL = 0
    BASIC = 1


class Uint16Vector(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsUint16Vector(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Uint16Vector()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Uint16VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # Uint16Vector
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Uint16Vector
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Uint16Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2),
            )
        return 0

    # Uint16Vector
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint16Flags, o)
        return 0

    # Uint16Vector
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Uint16Vector
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def Uint16VectorStart(builder):
    builder.StartObject(1)


def Uint16VectorAddValues(builder, values):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0
    )


def Uint16VectorStartValuesVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)


def Uint16VectorEnd(builder):
    return builder.EndObject()


class Uint16VectorT(object):

    # Uint16VectorT
    def __init__(self):
        self.values = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        uint16Vector = Uint16Vector()
        uint16Vector.Init(buf, pos)
        return cls.InitFromObj(uint16Vector)

    @classmethod
    def InitFromObj(cls, uint16Vector):
        x = Uint16VectorT()
        x._UnPack(uint16Vector)
        return x

    # Uint16VectorT
    def _UnPack(self, uint16Vector):
        if uint16Vector is None:
            return
        if not uint16Vector.ValuesIsNone():
            if np is None:
                self.values = []
                for i in range(uint16Vector.ValuesLength()):
                    self.values.append(uint16Vector.Values(i))
            else:
                self.values = uint16Vector.ValuesAsNumpy()

    # Uint16VectorT
    def Pack(self, builder):
        if self.values is not None:
            if np is not None and type(self.values) is np.ndarray:
                values = builder.CreateNumpyVector(self.values)
            else:
                Uint16VectorStartValuesVector(builder, len(self.values))
                for i in reversed(range(len(self.values))):
                    builder.PrependUint16(self.values[i])
                values = builder.EndVector(len(self.values))
        Uint16VectorStart(builder)
        if self.values is not None:
            Uint16VectorAddValues(builder, values)
        uint16Vector = Uint16VectorEnd(builder)
        return uint16Vector


class LocalResponseNormalizationOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsLocalResponseNormalizationOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LocalResponseNormalizationOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LocalResponseNormalizationOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # LocalResponseNormalizationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LocalResponseNormalizationOptions
    def Radius(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # LocalResponseNormalizationOptions
    def Bias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # LocalResponseNormalizationOptions
    def Alpha(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # LocalResponseNormalizationOptions
    def Beta(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0


def LocalResponseNormalizationOptionsStart(builder):
    builder.StartObject(4)


def LocalResponseNormalizationOptionsAddRadius(builder, radius):
    builder.PrependInt32Slot(0, radius, 0)


def LocalResponseNormalizationOptionsAddBias(builder, bias):
    builder.PrependFloat32Slot(1, bias, 0.0)


def LocalResponseNormalizationOptionsAddAlpha(builder, alpha):
    builder.PrependFloat32Slot(2, alpha, 0.0)


def LocalResponseNormalizationOptionsAddBeta(builder, beta):
    builder.PrependFloat32Slot(3, beta, 0.0)


def LocalResponseNormalizationOptionsEnd(builder):
    return builder.EndObject()


class LocalResponseNormalizationOptionsT(object):

    # LocalResponseNormalizationOptionsT
    def __init__(self):
        self.radius = 0  # type: int
        self.bias = 0.0  # type: float
        self.alpha = 0.0  # type: float
        self.beta = 0.0  # type: float

    @classmethod
    def InitFromBuf(cls, buf, pos):
        localResponseNormalizationOptions = LocalResponseNormalizationOptions()
        localResponseNormalizationOptions.Init(buf, pos)
        return cls.InitFromObj(localResponseNormalizationOptions)

    @classmethod
    def InitFromObj(cls, localResponseNormalizationOptions):
        x = LocalResponseNormalizationOptionsT()
        x._UnPack(localResponseNormalizationOptions)
        return x

    # LocalResponseNormalizationOptionsT
    def _UnPack(self, localResponseNormalizationOptions):
        if localResponseNormalizationOptions is None:
            return
        self.radius = localResponseNormalizationOptions.Radius()
        self.bias = localResponseNormalizationOptions.Bias()
        self.alpha = localResponseNormalizationOptions.Alpha()
        self.beta = localResponseNormalizationOptions.Beta()

    # LocalResponseNormalizationOptionsT
    def Pack(self, builder):
        LocalResponseNormalizationOptionsStart(builder)
        LocalResponseNormalizationOptionsAddRadius(builder, self.radius)
        LocalResponseNormalizationOptionsAddBias(builder, self.bias)
        LocalResponseNormalizationOptionsAddAlpha(builder, self.alpha)
        LocalResponseNormalizationOptionsAddBeta(builder, self.beta)
        localResponseNormalizationOptions = LocalResponseNormalizationOptionsEnd(
            builder
        )
        return localResponseNormalizationOptions


class FullyConnectedOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsFullyConnectedOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FullyConnectedOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FullyConnectedOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # FullyConnectedOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FullyConnectedOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # FullyConnectedOptions
    def WeightsFormat(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # FullyConnectedOptions
    def KeepNumDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False

    # FullyConnectedOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def FullyConnectedOptionsStart(builder):
    builder.StartObject(4)


def FullyConnectedOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(0, fusedActivationFunction, 0)


def FullyConnectedOptionsAddWeightsFormat(builder, weightsFormat):
    builder.PrependInt8Slot(1, weightsFormat, 0)


def FullyConnectedOptionsAddKeepNumDims(builder, keepNumDims):
    builder.PrependBoolSlot(2, keepNumDims, 0)


def FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    builder.PrependBoolSlot(3, asymmetricQuantizeInputs, 0)


def FullyConnectedOptionsEnd(builder):
    return builder.EndObject()


class FullyConnectedOptionsT(object):

    # FullyConnectedOptionsT
    def __init__(self):
        self.fusedActivationFunction = 0  # type: int
        self.weightsFormat = 0  # type: int
        self.keepNumDims = False  # type: bool
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fullyConnectedOptions = FullyConnectedOptions()
        fullyConnectedOptions.Init(buf, pos)
        return cls.InitFromObj(fullyConnectedOptions)

    @classmethod
    def InitFromObj(cls, fullyConnectedOptions):
        x = FullyConnectedOptionsT()
        x._UnPack(fullyConnectedOptions)
        return x

    # FullyConnectedOptionsT
    def _UnPack(self, fullyConnectedOptions):
        if fullyConnectedOptions is None:
            return
        self.fusedActivationFunction = fullyConnectedOptions.FusedActivationFunction()
        self.weightsFormat = fullyConnectedOptions.WeightsFormat()
        self.keepNumDims = fullyConnectedOptions.KeepNumDims()
        self.asymmetricQuantizeInputs = fullyConnectedOptions.AsymmetricQuantizeInputs()

    # FullyConnectedOptionsT
    def Pack(self, builder):
        FullyConnectedOptionsStart(builder)
        FullyConnectedOptionsAddFusedActivationFunction(
            builder, self.fusedActivationFunction
        )
        FullyConnectedOptionsAddWeightsFormat(builder, self.weightsFormat)
        FullyConnectedOptionsAddKeepNumDims(builder, self.keepNumDims)
        FullyConnectedOptionsAddAsymmetricQuantizeInputs(
            builder, self.asymmetricQuantizeInputs
        )
        fullyConnectedOptions = FullyConnectedOptionsEnd(builder)
        return fullyConnectedOptions


class ReducerOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsReducerOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReducerOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ReducerOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ReducerOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReducerOptions
    def KeepDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
            )
        return False


def ReducerOptionsStart(builder):
    builder.StartObject(1)


def ReducerOptionsAddKeepDims(builder, keepDims):
    builder.PrependBoolSlot(0, keepDims, 0)


def ReducerOptionsEnd(builder):
    return builder.EndObject()


class ReducerOptionsT(object):

    # ReducerOptionsT
    def __init__(self):
        self.keepDims = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reducerOptions = ReducerOptions()
        reducerOptions.Init(buf, pos)
        return cls.InitFromObj(reducerOptions)

    @classmethod
    def InitFromObj(cls, reducerOptions):
        x = ReducerOptionsT()
        x._UnPack(reducerOptions)
        return x

    # ReducerOptionsT
    def _UnPack(self, reducerOptions):
        if reducerOptions is None:
            return
        self.keepDims = reducerOptions.KeepDims()

    # ReducerOptionsT
    def Pack(self, builder):
        ReducerOptionsStart(builder)
        ReducerOptionsAddKeepDims(builder, self.keepDims)
        reducerOptions = ReducerOptionsEnd(builder)
        return reducerOptions


class GreaterEqualOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsGreaterEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GreaterEqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GreaterEqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # GreaterEqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def GreaterEqualOptionsStart(builder):
    builder.StartObject(0)


def GreaterEqualOptionsEnd(builder):
    return builder.EndObject()


class GreaterEqualOptionsT(object):

    # GreaterEqualOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        greaterEqualOptions = GreaterEqualOptions()
        greaterEqualOptions.Init(buf, pos)
        return cls.InitFromObj(greaterEqualOptions)

    @classmethod
    def InitFromObj(cls, greaterEqualOptions):
        x = GreaterEqualOptionsT()
        x._UnPack(greaterEqualOptions)
        return x

    # GreaterEqualOptionsT
    def _UnPack(self, greaterEqualOptions):
        if greaterEqualOptions is None:
            return

    # GreaterEqualOptionsT
    def Pack(self, builder):
        GreaterEqualOptionsStart(builder)
        greaterEqualOptions = GreaterEqualOptionsEnd(builder)
        return greaterEqualOptions


class NonMaxSuppressionV5Options(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsNonMaxSuppressionV5Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NonMaxSuppressionV5Options()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NonMaxSuppressionV5OptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # NonMaxSuppressionV5Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NonMaxSuppressionV5OptionsStart(builder):
    builder.StartObject(0)


def NonMaxSuppressionV5OptionsEnd(builder):
    return builder.EndObject()


class NonMaxSuppressionV5OptionsT(object):

    # NonMaxSuppressionV5OptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        nonMaxSuppressionV5Options = NonMaxSuppressionV5Options()
        nonMaxSuppressionV5Options.Init(buf, pos)
        return cls.InitFromObj(nonMaxSuppressionV5Options)

    @classmethod
    def InitFromObj(cls, nonMaxSuppressionV5Options):
        x = NonMaxSuppressionV5OptionsT()
        x._UnPack(nonMaxSuppressionV5Options)
        return x

    # NonMaxSuppressionV5OptionsT
    def _UnPack(self, nonMaxSuppressionV5Options):
        if nonMaxSuppressionV5Options is None:
            return

    # NonMaxSuppressionV5OptionsT
    def Pack(self, builder):
        NonMaxSuppressionV5OptionsStart(builder)
        nonMaxSuppressionV5Options = NonMaxSuppressionV5OptionsEnd(builder)
        return nonMaxSuppressionV5Options


class SquareOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsSquareOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SquareOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SquareOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SquareOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def SquareOptionsStart(builder):
    builder.StartObject(0)


def SquareOptionsEnd(builder):
    return builder.EndObject()


class SquareOptionsT(object):

    # SquareOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        squareOptions = SquareOptions()
        squareOptions.Init(buf, pos)
        return cls.InitFromObj(squareOptions)

    @classmethod
    def InitFromObj(cls, squareOptions):
        x = SquareOptionsT()
        x._UnPack(squareOptions)
        return x

    # SquareOptionsT
    def _UnPack(self, squareOptions):
        if squareOptions is None:
            return

    # SquareOptionsT
    def Pack(self, builder):
        SquareOptionsStart(builder)
        squareOptions = SquareOptionsEnd(builder)
        return squareOptions


class GatherNdOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsGatherNdOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherNdOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GatherNdOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # GatherNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def GatherNdOptionsStart(builder):
    builder.StartObject(0)


def GatherNdOptionsEnd(builder):
    return builder.EndObject()


class GatherNdOptionsT(object):

    # GatherNdOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        gatherNdOptions = GatherNdOptions()
        gatherNdOptions.Init(buf, pos)
        return cls.InitFromObj(gatherNdOptions)

    @classmethod
    def InitFromObj(cls, gatherNdOptions):
        x = GatherNdOptionsT()
        x._UnPack(gatherNdOptions)
        return x

    # GatherNdOptionsT
    def _UnPack(self, gatherNdOptions):
        if gatherNdOptions is None:
            return

    # GatherNdOptionsT
    def Pack(self, builder):
        GatherNdOptionsStart(builder)
        gatherNdOptions = GatherNdOptionsEnd(builder)
        return gatherNdOptions
