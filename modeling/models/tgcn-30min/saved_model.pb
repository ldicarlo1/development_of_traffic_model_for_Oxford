¿
¼
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12v2.4.0-49-g85c8b2a817f8Î
¨
&fixed_adjacency_graph_convolution_13/AVarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*7
shared_name(&fixed_adjacency_graph_convolution_13/A
¡
:fixed_adjacency_graph_convolution_13/A/Read/ReadVariableOpReadVariableOp&fixed_adjacency_graph_convolution_13/A*
_output_shapes

:FF*
dtype0
²
+fixed_adjacency_graph_convolution_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+fixed_adjacency_graph_convolution_13/kernel
«
?fixed_adjacency_graph_convolution_13/kernel/Read/ReadVariableOpReadVariableOp+fixed_adjacency_graph_convolution_13/kernel*
_output_shapes

:*
dtype0
®
)fixed_adjacency_graph_convolution_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*:
shared_name+)fixed_adjacency_graph_convolution_13/bias
§
=fixed_adjacency_graph_convolution_13/bias/Read/ReadVariableOpReadVariableOp)fixed_adjacency_graph_convolution_13/bias*
_output_shapes

:F*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:dF*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:F*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F*,
shared_namelstm_13/lstm_cell_13/kernel

/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel*
_output_shapes
:	F*
dtype0
§
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
 
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_13/lstm_cell_13/bias

-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
À
2Adam/fixed_adjacency_graph_convolution_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/fixed_adjacency_graph_convolution_13/kernel/m
¹
FAdam/fixed_adjacency_graph_convolution_13/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/fixed_adjacency_graph_convolution_13/kernel/m*
_output_shapes

:*
dtype0
¼
0Adam/fixed_adjacency_graph_convolution_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*A
shared_name20Adam/fixed_adjacency_graph_convolution_13/bias/m
µ
DAdam/fixed_adjacency_graph_convolution_13/bias/m/Read/ReadVariableOpReadVariableOp0Adam/fixed_adjacency_graph_convolution_13/bias/m*
_output_shapes

:F*
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF*'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:dF*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:F*
dtype0
¡
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m

6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m*
_output_shapes
:	F*
dtype0
µ
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
®
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m*
_output_shapes
:	d*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m

4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:*
dtype0
À
2Adam/fixed_adjacency_graph_convolution_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/fixed_adjacency_graph_convolution_13/kernel/v
¹
FAdam/fixed_adjacency_graph_convolution_13/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/fixed_adjacency_graph_convolution_13/kernel/v*
_output_shapes

:*
dtype0
¼
0Adam/fixed_adjacency_graph_convolution_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*A
shared_name20Adam/fixed_adjacency_graph_convolution_13/bias/v
µ
DAdam/fixed_adjacency_graph_convolution_13/bias/v/Read/ReadVariableOpReadVariableOp0Adam/fixed_adjacency_graph_convolution_13/bias/v*
_output_shapes

:F*
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF*'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:dF*
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:F*
dtype0
¡
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v

6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v*
_output_shapes
:	F*
dtype0
µ
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
®
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v*
_output_shapes
:	d*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v

4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ë:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¦:
value:B: B:
Á
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer_with_weights-2

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 

	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
o
A

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
l
)cell
*
state_spec
+regularization_losses
,trainable_variables
-	variables
.	keras_api
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
Ì
9iter

:beta_1

;beta_2
	<decay
=learning_ratemm3m4m>m?m@mvv3v4v>v?v@v
 
1
0
1
>2
?3
@4
35
46
8
0
1
2
>3
?4
@5
36
47
­
regularization_losses

Alayers
trainable_variables
Bnon_trainable_variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
	variables
 
 
 
 
 
­
regularization_losses

Flayers
trainable_variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
	variables
mk
VARIABLE_VALUE&fixed_adjacency_graph_convolution_13/A1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE+fixed_adjacency_graph_convolution_13/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE)fixed_adjacency_graph_convolution_13/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
­
regularization_losses

Klayers
trainable_variables
Lnon_trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
	variables
 
 
 
­
regularization_losses

Players
trainable_variables
Qnon_trainable_variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
	variables
 
 
 
­
!regularization_losses

Ulayers
"trainable_variables
Vnon_trainable_variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
#	variables
 
 
 
­
%regularization_losses

Zlayers
&trainable_variables
[non_trainable_variables
\layer_regularization_losses
]metrics
^layer_metrics
'	variables
~

>kernel
?recurrent_kernel
@bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
 
 

>0
?1
@2

>0
?1
@2
¹
+regularization_losses

clayers
,trainable_variables
dnon_trainable_variables
elayer_regularization_losses
fmetrics
glayer_metrics
-	variables

hstates
 
 
 
­
/regularization_losses

ilayers
0trainable_variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
1	variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
­
5regularization_losses

nlayers
6trainable_variables
onon_trainable_variables
player_regularization_losses
qmetrics
rlayer_metrics
7	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_13/lstm_cell_13/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_13/lstm_cell_13/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_13/lstm_cell_13/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
F
0
1
2
3
4
5
6
7
	8

9

0
 

s0
t1
 
 
 
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1
@2

>0
?1
@2
­
_regularization_losses

ulayers
`trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
ylayer_metrics
a	variables

)0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ztotal
	{count
|	variables
}	keras_api
G
	~total
	count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

|	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

	variables

VARIABLE_VALUE2Adam/fixed_adjacency_graph_convolution_13/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/fixed_adjacency_graph_convolution_13/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/fixed_adjacency_graph_convolution_13/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/fixed_adjacency_graph_convolution_13/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_27Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿF
Â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_27&fixed_adjacency_graph_convolution_13/A+fixed_adjacency_graph_convolution_13/kernel)fixed_adjacency_graph_convolution_13/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_13/kerneldense_13/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_235460
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:fixed_adjacency_graph_convolution_13/A/Read/ReadVariableOp?fixed_adjacency_graph_convolution_13/kernel/Read/ReadVariableOp=fixed_adjacency_graph_convolution_13/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpFAdam/fixed_adjacency_graph_convolution_13/kernel/m/Read/ReadVariableOpDAdam/fixed_adjacency_graph_convolution_13/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpFAdam/fixed_adjacency_graph_convolution_13/kernel/v/Read/ReadVariableOpDAdam/fixed_adjacency_graph_convolution_13/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_237031
©	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&fixed_adjacency_graph_convolution_13/A+fixed_adjacency_graph_convolution_13/kernel)fixed_adjacency_graph_convolution_13/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biastotalcounttotal_1count_12Adam/fixed_adjacency_graph_convolution_13/kernel/m0Adam/fixed_adjacency_graph_convolution_13/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m"Adam/lstm_13/lstm_cell_13/kernel/m,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m Adam/lstm_13/lstm_cell_13/bias/m2Adam/fixed_adjacency_graph_convolution_13/kernel/v0Adam/fixed_adjacency_graph_convolution_13/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v"Adam/lstm_13/lstm_cell_13/kernel/v,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v Adam/lstm_13/lstm_cell_13/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_237134Ü°
¢
d
+__inference_dropout_13_layer_call_fn_236790

inputs
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
»
Í
-__inference_lstm_cell_13_layer_call_fn_236915

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
[
ò
C__inference_lstm_13_layer_call_and_return_conditional_losses_236265

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_236180*
condR
while_cond_236179*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
«

"model_13_lstm_13_while_cond_234042>
:model_13_lstm_13_while_model_13_lstm_13_while_loop_counterD
@model_13_lstm_13_while_model_13_lstm_13_while_maximum_iterations&
"model_13_lstm_13_while_placeholder(
$model_13_lstm_13_while_placeholder_1(
$model_13_lstm_13_while_placeholder_2(
$model_13_lstm_13_while_placeholder_3@
<model_13_lstm_13_while_less_model_13_lstm_13_strided_slice_1V
Rmodel_13_lstm_13_while_model_13_lstm_13_while_cond_234042___redundant_placeholder0V
Rmodel_13_lstm_13_while_model_13_lstm_13_while_cond_234042___redundant_placeholder1V
Rmodel_13_lstm_13_while_model_13_lstm_13_while_cond_234042___redundant_placeholder2V
Rmodel_13_lstm_13_while_model_13_lstm_13_while_cond_234042___redundant_placeholder3#
model_13_lstm_13_while_identity
Å
model_13/lstm_13/while/LessLess"model_13_lstm_13_while_placeholder<model_13_lstm_13_while_less_model_13_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
model_13/lstm_13/while/Less
model_13/lstm_13/while/IdentityIdentitymodel_13/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2!
model_13/lstm_13/while/Identity"K
model_13_lstm_13_while_identity(model_13/lstm_13/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ð	
Ý
D__inference_dense_13_layer_call_and_return_conditional_losses_235279

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
«
Ã
while_cond_236507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236507___redundant_placeholder04
0while_while_cond_236507___redundant_placeholder14
0while_while_cond_236507___redundant_placeholder24
0while_while_cond_236507___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
É
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_236785

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
C
þ
while_body_236508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 


(__inference_lstm_13_layer_call_fn_236757
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2346182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
«
Ã
while_cond_236660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236660___redundant_placeholder04
0while_while_cond_236660___redundant_placeholder14
0while_while_cond_236660___redundant_placeholder24
0while_while_cond_236660___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
»
Í
-__inference_lstm_cell_13_layer_call_fn_236898

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
Ç[
ô
C__inference_lstm_13_layer_call_and_return_conditional_losses_236746
inputs_0/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_236661*
condR
while_cond_236660*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
ð	
Ý
D__inference_dense_13_layer_call_and_return_conditional_losses_236806

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
³
Ý
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236848

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
'
â
D__inference_model_13_layer_call_and_return_conditional_losses_235326
input_27/
+fixed_adjacency_graph_convolution_13_235302/
+fixed_adjacency_graph_convolution_13_235304/
+fixed_adjacency_graph_convolution_13_235306
lstm_13_235312
lstm_13_235314
lstm_13_235316
dense_13_235320
dense_13_235322
identity¢ dense_13/StatefulPartitionedCall¢<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim¹
tf.expand_dims_13/ExpandDims
ExpandDimsinput_27)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsý
reshape_39/PartitionedCallPartitionedCall%tf.expand_dims_13/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_39_layer_call_and_return_conditional_losses_2347782
reshape_39/PartitionedCallð
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_39/PartitionedCall:output:0+fixed_adjacency_graph_convolution_13_235302+fixed_adjacency_graph_convolution_13_235304+fixed_adjacency_graph_convolution_13_235306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_2348392>
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¡
reshape_40/PartitionedCallPartitionedCallEfixed_adjacency_graph_convolution_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_40_layer_call_and_return_conditional_losses_2348732
reshape_40/PartitionedCallÿ
permute_13/PartitionedCallPartitionedCall#reshape_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_permute_13_layer_call_and_return_conditional_losses_2341432
permute_13/PartitionedCallû
reshape_41/PartitionedCallPartitionedCall#permute_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_41_layer_call_and_return_conditional_losses_2348952
reshape_41/PartitionedCall¾
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_41/PartitionedCall:output:0lstm_13_235312lstm_13_235314lstm_13_235316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2352082!
lstm_13/StatefulPartitionedCallü
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352552
dropout_13/PartitionedCall±
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_235320dense_13_235322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2352792"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall=^fixed_adjacency_graph_convolution_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2|
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
«
Ú
)__inference_model_13_layer_call_fn_235429
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_2354102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
À(

D__inference_model_13_layer_call_and_return_conditional_losses_235296
input_27/
+fixed_adjacency_graph_convolution_13_234852/
+fixed_adjacency_graph_convolution_13_234854/
+fixed_adjacency_graph_convolution_13_234856
lstm_13_235231
lstm_13_235233
lstm_13_235235
dense_13_235290
dense_13_235292
identity¢ dense_13/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim¹
tf.expand_dims_13/ExpandDims
ExpandDimsinput_27)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsý
reshape_39/PartitionedCallPartitionedCall%tf.expand_dims_13/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_39_layer_call_and_return_conditional_losses_2347782
reshape_39/PartitionedCallð
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_39/PartitionedCall:output:0+fixed_adjacency_graph_convolution_13_234852+fixed_adjacency_graph_convolution_13_234854+fixed_adjacency_graph_convolution_13_234856*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_2348392>
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¡
reshape_40/PartitionedCallPartitionedCallEfixed_adjacency_graph_convolution_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_40_layer_call_and_return_conditional_losses_2348732
reshape_40/PartitionedCallÿ
permute_13/PartitionedCallPartitionedCall#reshape_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_permute_13_layer_call_and_return_conditional_losses_2341432
permute_13/PartitionedCallû
reshape_41/PartitionedCallPartitionedCall#permute_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_41_layer_call_and_return_conditional_losses_2348952
reshape_41/PartitionedCall¾
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_41/PartitionedCall:output:0lstm_13_235231lstm_13_235233lstm_13_235235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2350552!
lstm_13/StatefulPartitionedCall
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352502$
"dropout_13/StatefulPartitionedCall¹
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_235290dense_13_235292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2352792"
 dense_13/StatefulPartitionedCall¦
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall=^fixed_adjacency_graph_convolution_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2|
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
³
Ý
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236881

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
O
þ	
lstm_13_while_body_235609,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0A
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0@
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor=
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource?
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource>
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpÓ
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemá
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp÷
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_13/while/lstm_cell_13/MatMulç
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpà
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/while/lstm_cell_13/MatMul_1Ø
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_13/while/lstm_cell_13/addà
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpå
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_13/while/lstm_cell_13/BiasAdd
 lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_13/while/lstm_cell_13/Const
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim«
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2"
 lstm_13/while/lstm_cell_13/split°
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"lstm_13/while/lstm_cell_13/Sigmoid´
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_13/while/lstm_cell_13/Sigmoid_1À
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/while/lstm_cell_13/mul§
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
lstm_13/while/lstm_cell_13/ReluÔ
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/mul_1É
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/add_1´
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_13/while/lstm_cell_13/Sigmoid_2¦
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!lstm_13/while/lstm_cell_13/Relu_1Ø
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/mul_2
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity­
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1
lstm_13/while/Identity_2Identitylstm_13/while/add:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2Á
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3´
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/while/Identity_4´
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/while/Identity_5"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
£
G
+__inference_permute_13_layer_call_fn_234149

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_permute_13_layer_call_and_return_conditional_losses_2341432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
b
F__inference_reshape_40_layer_call_and_return_conditional_losses_234873

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
«
Ã
while_cond_236332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236332___redundant_placeholder04
0while_while_cond_236332___redundant_placeholder14
0while_while_cond_236332___redundant_placeholder24
0while_while_cond_236332___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ÛD
Ü
C__inference_lstm_13_layer_call_and_return_conditional_losses_234750

inputs
lstm_cell_13_234668
lstm_cell_13_234670
lstm_cell_13_234672
identity¢$lstm_cell_13/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_234668lstm_cell_13_234670lstm_cell_13_234672*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342552&
$lstm_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_234668lstm_cell_13_234670lstm_cell_13_234672*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_234681*
condR
while_cond_234680*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_13/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
I
¡
__inference__traced_save_237031
file_prefixE
Asavev2_fixed_adjacency_graph_convolution_13_a_read_readvariableopJ
Fsavev2_fixed_adjacency_graph_convolution_13_kernel_read_readvariableopH
Dsavev2_fixed_adjacency_graph_convolution_13_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopQ
Msavev2_adam_fixed_adjacency_graph_convolution_13_kernel_m_read_readvariableopO
Ksavev2_adam_fixed_adjacency_graph_convolution_13_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableopQ
Msavev2_adam_fixed_adjacency_graph_convolution_13_kernel_v_read_readvariableopO
Ksavev2_adam_fixed_adjacency_graph_convolution_13_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameï
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value÷Bô B1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_fixed_adjacency_graph_convolution_13_a_read_readvariableopFsavev2_fixed_adjacency_graph_convolution_13_kernel_read_readvariableopDsavev2_fixed_adjacency_graph_convolution_13_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopMsavev2_adam_fixed_adjacency_graph_convolution_13_kernel_m_read_readvariableopKsavev2_adam_fixed_adjacency_graph_convolution_13_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableopMsavev2_adam_fixed_adjacency_graph_convolution_13_kernel_v_read_readvariableopKsavev2_adam_fixed_adjacency_graph_convolution_13_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ø
_input_shapesæ
ã: :FF::F:dF:F: : : : : :	F:	d:: : : : ::F:dF:F:	F:	d:::F:dF:F:	F:	d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:FF:$ 

_output_shapes

::$ 

_output_shapes

:F:$ 

_output_shapes

:dF: 

_output_shapes
:F:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	F:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

:F:$ 

_output_shapes

:dF: 

_output_shapes
:F:%!

_output_shapes
:	F:%!

_output_shapes
:	d:!

_output_shapes	
::$ 

_output_shapes

::$ 

_output_shapes

:F:$ 

_output_shapes

:dF: 

_output_shapes
:F:%!

_output_shapes
:	F:%!

_output_shapes
:	d:!

_output_shapes	
:: 

_output_shapes
: 
O
þ	
lstm_13_while_body_235858,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0A
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0@
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor=
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource?
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource>
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpÓ
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemá
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp÷
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_13/while/lstm_cell_13/MatMulç
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpà
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/while/lstm_cell_13/MatMul_1Ø
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_13/while/lstm_cell_13/addà
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpå
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_13/while/lstm_cell_13/BiasAdd
 lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_13/while/lstm_cell_13/Const
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim«
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2"
 lstm_13/while/lstm_cell_13/split°
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"lstm_13/while/lstm_cell_13/Sigmoid´
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_13/while/lstm_cell_13/Sigmoid_1À
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/while/lstm_cell_13/mul§
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
lstm_13/while/lstm_cell_13/ReluÔ
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/mul_1É
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/add_1´
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$lstm_13/while/lstm_cell_13/Sigmoid_2¦
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!lstm_13/while/lstm_cell_13/Relu_1Ø
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 lstm_13/while/lstm_cell_13/mul_2
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity­
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1
lstm_13/while/Identity_2Identitylstm_13/while/add:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2Á
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3´
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/while/Identity_4´
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:02^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/while/Identity_5"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
«
Ã
while_cond_235122
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_235122___redundant_placeholder04
0while_while_cond_235122___redundant_placeholder14
0while_while_cond_235122___redundant_placeholder24
0while_while_cond_235122___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
%

while_body_234681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_13_234705_0
while_lstm_cell_13_234707_0
while_lstm_cell_13_234709_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_13_234705
while_lstm_cell_13_234707
while_lstm_cell_13_234709¢*while/lstm_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_234705_0while_lstm_cell_13_234707_0while_lstm_cell_13_234709_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342552,
*while/lstm_cell_13/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2º
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ä
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1+^while/lstm_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4Ä
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2+^while/lstm_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_234705while_lstm_cell_13_234705_0"8
while_lstm_cell_13_234707while_lstm_cell_13_234707_0"8
while_lstm_cell_13_234709while_lstm_cell_13_234709_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
À,
»
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_234839
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity¢add/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_5
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
add®
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features
Ç[
ô
C__inference_lstm_13_layer_call_and_return_conditional_losses_236593
inputs_0/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_236508*
condR
while_cond_236507*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0

Õ
$__inference_signature_wrapper_235460
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_2341362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
ï
b
F__inference_reshape_39_layer_call_and_return_conditional_losses_234778

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¼\

"model_13_lstm_13_while_body_234043>
:model_13_lstm_13_while_model_13_lstm_13_while_loop_counterD
@model_13_lstm_13_while_model_13_lstm_13_while_maximum_iterations&
"model_13_lstm_13_while_placeholder(
$model_13_lstm_13_while_placeholder_1(
$model_13_lstm_13_while_placeholder_2(
$model_13_lstm_13_while_placeholder_3=
9model_13_lstm_13_while_model_13_lstm_13_strided_slice_1_0y
umodel_13_lstm_13_while_tensorarrayv2read_tensorlistgetitem_model_13_lstm_13_tensorarrayunstack_tensorlistfromtensor_0H
Dmodel_13_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0J
Fmodel_13_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0I
Emodel_13_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0#
model_13_lstm_13_while_identity%
!model_13_lstm_13_while_identity_1%
!model_13_lstm_13_while_identity_2%
!model_13_lstm_13_while_identity_3%
!model_13_lstm_13_while_identity_4%
!model_13_lstm_13_while_identity_5;
7model_13_lstm_13_while_model_13_lstm_13_strided_slice_1w
smodel_13_lstm_13_while_tensorarrayv2read_tensorlistgetitem_model_13_lstm_13_tensorarrayunstack_tensorlistfromtensorF
Bmodel_13_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceH
Dmodel_13_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceG
Cmodel_13_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource¢:model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢9model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢;model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpå
Hmodel_13/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2J
Hmodel_13/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape¹
:model_13/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemumodel_13_lstm_13_while_tensorarrayv2read_tensorlistgetitem_model_13_lstm_13_tensorarrayunstack_tensorlistfromtensor_0"model_13_lstm_13_while_placeholderQmodel_13/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02<
:model_13/lstm_13/while/TensorArrayV2Read/TensorListGetItemü
9model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpDmodel_13_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02;
9model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp
*model_13/lstm_13/while/lstm_cell_13/MatMulMatMulAmodel_13/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Amodel_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_13/lstm_13/while/lstm_cell_13/MatMul
;model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpFmodel_13_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02=
;model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp
,model_13/lstm_13/while/lstm_cell_13/MatMul_1MatMul$model_13_lstm_13_while_placeholder_2Cmodel_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_13/lstm_13/while/lstm_cell_13/MatMul_1ü
'model_13/lstm_13/while/lstm_cell_13/addAddV24model_13/lstm_13/while/lstm_cell_13/MatMul:product:06model_13/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'model_13/lstm_13/while/lstm_cell_13/addû
:model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpEmodel_13_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp
+model_13/lstm_13/while/lstm_cell_13/BiasAddBiasAdd+model_13/lstm_13/while/lstm_cell_13/add:z:0Bmodel_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+model_13/lstm_13/while/lstm_cell_13/BiasAdd
)model_13/lstm_13/while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_13/lstm_13/while/lstm_cell_13/Const¬
3model_13/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_13/lstm_13/while/lstm_cell_13/split/split_dimÏ
)model_13/lstm_13/while/lstm_cell_13/splitSplit<model_13/lstm_13/while/lstm_cell_13/split/split_dim:output:04model_13/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2+
)model_13/lstm_13/while/lstm_cell_13/splitË
+model_13/lstm_13/while/lstm_cell_13/SigmoidSigmoid2model_13/lstm_13/while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2-
+model_13/lstm_13/while/lstm_cell_13/SigmoidÏ
-model_13/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid2model_13/lstm_13/while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2/
-model_13/lstm_13/while/lstm_cell_13/Sigmoid_1ä
'model_13/lstm_13/while/lstm_cell_13/mulMul1model_13/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0$model_13_lstm_13_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'model_13/lstm_13/while/lstm_cell_13/mulÂ
(model_13/lstm_13/while/lstm_cell_13/ReluRelu2model_13/lstm_13/while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(model_13/lstm_13/while/lstm_cell_13/Reluø
)model_13/lstm_13/while/lstm_cell_13/mul_1Mul/model_13/lstm_13/while/lstm_cell_13/Sigmoid:y:06model_13/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2+
)model_13/lstm_13/while/lstm_cell_13/mul_1í
)model_13/lstm_13/while/lstm_cell_13/add_1AddV2+model_13/lstm_13/while/lstm_cell_13/mul:z:0-model_13/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2+
)model_13/lstm_13/while/lstm_cell_13/add_1Ï
-model_13/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid2model_13/lstm_13/while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2/
-model_13/lstm_13/while/lstm_cell_13/Sigmoid_2Á
*model_13/lstm_13/while/lstm_cell_13/Relu_1Relu-model_13/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2,
*model_13/lstm_13/while/lstm_cell_13/Relu_1ü
)model_13/lstm_13/while/lstm_cell_13/mul_2Mul1model_13/lstm_13/while/lstm_cell_13/Sigmoid_2:y:08model_13/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2+
)model_13/lstm_13/while/lstm_cell_13/mul_2µ
;model_13/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$model_13_lstm_13_while_placeholder_1"model_13_lstm_13_while_placeholder-model_13/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;model_13/lstm_13/while/TensorArrayV2Write/TensorListSetItem~
model_13/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_13/lstm_13/while/add/y­
model_13/lstm_13/while/addAddV2"model_13_lstm_13_while_placeholder%model_13/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/while/add
model_13/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
model_13/lstm_13/while/add_1/yË
model_13/lstm_13/while/add_1AddV2:model_13_lstm_13_while_model_13_lstm_13_while_loop_counter'model_13/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/while/add_1È
model_13/lstm_13/while/IdentityIdentity model_13/lstm_13/while/add_1:z:0;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_13/lstm_13/while/Identityì
!model_13/lstm_13/while/Identity_1Identity@model_13_lstm_13_while_model_13_lstm_13_while_maximum_iterations;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!model_13/lstm_13/while/Identity_1Ê
!model_13/lstm_13/while/Identity_2Identitymodel_13/lstm_13/while/add:z:0;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!model_13/lstm_13/while/Identity_2÷
!model_13/lstm_13/while/Identity_3IdentityKmodel_13/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!model_13/lstm_13/while/Identity_3ê
!model_13/lstm_13/while/Identity_4Identity-model_13/lstm_13/while/lstm_cell_13/mul_2:z:0;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!model_13/lstm_13/while/Identity_4ê
!model_13/lstm_13/while/Identity_5Identity-model_13/lstm_13/while/lstm_cell_13/add_1:z:0;^model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:^model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp<^model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!model_13/lstm_13/while/Identity_5"K
model_13_lstm_13_while_identity(model_13/lstm_13/while/Identity:output:0"O
!model_13_lstm_13_while_identity_1*model_13/lstm_13/while/Identity_1:output:0"O
!model_13_lstm_13_while_identity_2*model_13/lstm_13/while/Identity_2:output:0"O
!model_13_lstm_13_while_identity_3*model_13/lstm_13/while/Identity_3:output:0"O
!model_13_lstm_13_while_identity_4*model_13/lstm_13/while/Identity_4:output:0"O
!model_13_lstm_13_while_identity_5*model_13/lstm_13/while/Identity_5:output:0"
Cmodel_13_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceEmodel_13_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"
Dmodel_13_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceFmodel_13_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"
Bmodel_13_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceDmodel_13_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"t
7model_13_lstm_13_while_model_13_lstm_13_strided_slice_19model_13_lstm_13_while_model_13_lstm_13_strided_slice_1_0"ì
smodel_13_lstm_13_while_tensorarrayv2read_tensorlistgetitem_model_13_lstm_13_tensorarrayunstack_tensorlistfromtensorumodel_13_lstm_13_while_tensorarrayv2read_tensorlistgetitem_model_13_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2x
:model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:model_13/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2v
9model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp9model_13/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2z
;model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp;model_13/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
C
þ
while_body_234970
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_236780

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ø
b
F__inference_reshape_40_layer_call_and_return_conditional_losses_236089

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1m
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¥
Ø
)__inference_model_13_layer_call_fn_235972

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_2353592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs


(__inference_lstm_13_layer_call_fn_236768
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2347502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
inputs/0
«
Ã
while_cond_234969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234969___redundant_placeholder04
0while_while_cond_234969___redundant_placeholder14
0while_while_cond_234969___redundant_placeholder24
0while_while_cond_234969___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
[
ò
C__inference_lstm_13_layer_call_and_return_conditional_losses_235055

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_234970*
condR
while_cond_234969*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ø
b
F__inference_reshape_41_layer_call_and_return_conditional_losses_236107

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
«
Ú
)__inference_model_13_layer_call_fn_235378
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_2353592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
«
Û
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_234222

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
«
Ã
while_cond_234548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234548___redundant_placeholder04
0while_while_cond_234548___redundant_placeholder14
0while_while_cond_234548___redundant_placeholder24
0while_while_cond_234548___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
®
G
+__inference_reshape_41_layer_call_fn_236112

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_41_layer_call_and_return_conditional_losses_2348952
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
%

while_body_234549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_13_234573_0
while_lstm_cell_13_234575_0
while_lstm_cell_13_234577_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_13_234573
while_lstm_cell_13_234575
while_lstm_cell_13_234577¢*while/lstm_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_234573_0while_lstm_cell_13_234575_0while_lstm_cell_13_234577_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342222,
*while/lstm_cell_13/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2º
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_13/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ä
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1+^while/lstm_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4Ä
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2+^while/lstm_cell_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_13_234573while_lstm_cell_13_234573_0"8
while_lstm_cell_13_234575while_lstm_cell_13_234575_0"8
while_lstm_cell_13_234577while_lstm_cell_13_234577_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
[
ò
C__inference_lstm_13_layer_call_and_return_conditional_losses_236418

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_236333*
condR
while_cond_236332*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¥
Ø
)__inference_model_13_layer_call_fn_235993

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_2354102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs

G
+__inference_dropout_13_layer_call_fn_236795

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
à
b
F__inference_permute_13_layer_call_and_return_conditional_losses_234143

inputs
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transpose
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
'
à
D__inference_model_13_layer_call_and_return_conditional_losses_235410

inputs/
+fixed_adjacency_graph_convolution_13_235386/
+fixed_adjacency_graph_convolution_13_235388/
+fixed_adjacency_graph_convolution_13_235390
lstm_13_235396
lstm_13_235398
lstm_13_235400
dense_13_235404
dense_13_235406
identity¢ dense_13/StatefulPartitionedCall¢<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim·
tf.expand_dims_13/ExpandDims
ExpandDimsinputs)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsý
reshape_39/PartitionedCallPartitionedCall%tf.expand_dims_13/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_39_layer_call_and_return_conditional_losses_2347782
reshape_39/PartitionedCallð
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_39/PartitionedCall:output:0+fixed_adjacency_graph_convolution_13_235386+fixed_adjacency_graph_convolution_13_235388+fixed_adjacency_graph_convolution_13_235390*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_2348392>
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¡
reshape_40/PartitionedCallPartitionedCallEfixed_adjacency_graph_convolution_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_40_layer_call_and_return_conditional_losses_2348732
reshape_40/PartitionedCallÿ
permute_13/PartitionedCallPartitionedCall#reshape_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_permute_13_layer_call_and_return_conditional_losses_2341432
permute_13/PartitionedCallû
reshape_41/PartitionedCallPartitionedCall#permute_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_41_layer_call_and_return_conditional_losses_2348952
reshape_41/PartitionedCall¾
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_41/PartitionedCall:output:0lstm_13_235396lstm_13_235398lstm_13_235400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2352082!
lstm_13/StatefulPartitionedCallü
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352552
dropout_13/PartitionedCall±
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_235404dense_13_235406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2352792"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall=^fixed_adjacency_graph_convolution_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2|
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs


(__inference_lstm_13_layer_call_fn_236440

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2352082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ë
«
E__inference_fixed_adjacency_graph_convolution_13_layer_call_fn_236075
features
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallfeaturesunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_2348392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_235250

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
C
þ
while_body_235123
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
®
G
+__inference_reshape_40_layer_call_fn_236094

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_40_layer_call_and_return_conditional_losses_2348732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
C
þ
while_body_236180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 


ã
lstm_13_while_cond_235608,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1D
@lstm_13_while_lstm_13_while_cond_235608___redundant_placeholder0D
@lstm_13_while_lstm_13_while_cond_235608___redundant_placeholder1D
@lstm_13_while_lstm_13_while_cond_235608___redundant_placeholder2D
@lstm_13_while_lstm_13_while_cond_235608___redundant_placeholder3
lstm_13_while_identity

lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
C
þ
while_body_236661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ñ
Ù
!__inference__wrapped_model_234136
input_27Q
Mmodel_13_fixed_adjacency_graph_convolution_13_shape_1_readvariableop_resourceQ
Mmodel_13_fixed_adjacency_graph_convolution_13_shape_3_readvariableop_resourceM
Imodel_13_fixed_adjacency_graph_convolution_13_add_readvariableop_resource@
<model_13_lstm_13_lstm_cell_13_matmul_readvariableop_resourceB
>model_13_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceA
=model_13_lstm_13_lstm_cell_13_biasadd_readvariableop_resource4
0model_13_dense_13_matmul_readvariableop_resource5
1model_13_dense_13_biasadd_readvariableop_resource
identity¢(model_13/dense_13/BiasAdd/ReadVariableOp¢'model_13/dense_13/MatMul/ReadVariableOp¢@model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOp¢Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp¢Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp¢4model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢3model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢5model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢model_13/lstm_13/while¡
)model_13/tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_13/tf.expand_dims_13/ExpandDims/dimÔ
%model_13/tf.expand_dims_13/ExpandDims
ExpandDimsinput_272model_13/tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2'
%model_13/tf.expand_dims_13/ExpandDims
model_13/reshape_39/ShapeShape.model_13/tf.expand_dims_13/ExpandDims:output:0*
T0*
_output_shapes
:2
model_13/reshape_39/Shape
'model_13/reshape_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_13/reshape_39/strided_slice/stack 
)model_13/reshape_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_39/strided_slice/stack_1 
)model_13/reshape_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_39/strided_slice/stack_2Ú
!model_13/reshape_39/strided_sliceStridedSlice"model_13/reshape_39/Shape:output:00model_13/reshape_39/strided_slice/stack:output:02model_13/reshape_39/strided_slice/stack_1:output:02model_13/reshape_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_13/reshape_39/strided_slice
#model_13/reshape_39/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_13/reshape_39/Reshape/shape/1
#model_13/reshape_39/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_39/Reshape/shape/2
!model_13/reshape_39/Reshape/shapePack*model_13/reshape_39/strided_slice:output:0,model_13/reshape_39/Reshape/shape/1:output:0,model_13/reshape_39/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_13/reshape_39/Reshape/shape×
model_13/reshape_39/ReshapeReshape.model_13/tf.expand_dims_13/ExpandDims:output:0*model_13/reshape_39/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/reshape_39/ReshapeÑ
<model_13/fixed_adjacency_graph_convolution_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_13/fixed_adjacency_graph_convolution_13/transpose/perm¢
7model_13/fixed_adjacency_graph_convolution_13/transpose	Transpose$model_13/reshape_39/Reshape:output:0Emodel_13/fixed_adjacency_graph_convolution_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF29
7model_13/fixed_adjacency_graph_convolution_13/transposeÕ
3model_13/fixed_adjacency_graph_convolution_13/ShapeShape;model_13/fixed_adjacency_graph_convolution_13/transpose:y:0*
T0*
_output_shapes
:25
3model_13/fixed_adjacency_graph_convolution_13/Shapeæ
5model_13/fixed_adjacency_graph_convolution_13/unstackUnpack<model_13/fixed_adjacency_graph_convolution_13/Shape:output:0*
T0*
_output_shapes
: : : *	
num27
5model_13/fixed_adjacency_graph_convolution_13/unstack
Dmodel_13/fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOpReadVariableOpMmodel_13_fixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02F
Dmodel_13/fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOp¿
5model_13/fixed_adjacency_graph_convolution_13/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   27
5model_13/fixed_adjacency_graph_convolution_13/Shape_1ê
7model_13/fixed_adjacency_graph_convolution_13/unstack_1Unpack>model_13/fixed_adjacency_graph_convolution_13/Shape_1:output:0*
T0*
_output_shapes
: : *	
num29
7model_13/fixed_adjacency_graph_convolution_13/unstack_1Ë
;model_13/fixed_adjacency_graph_convolution_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2=
;model_13/fixed_adjacency_graph_convolution_13/Reshape/shape®
5model_13/fixed_adjacency_graph_convolution_13/ReshapeReshape;model_13/fixed_adjacency_graph_convolution_13/transpose:y:0Dmodel_13/fixed_adjacency_graph_convolution_13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF27
5model_13/fixed_adjacency_graph_convolution_13/Reshape¢
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpReadVariableOpMmodel_13_fixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02J
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpÑ
>model_13/fixed_adjacency_graph_convolution_13/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2@
>model_13/fixed_adjacency_graph_convolution_13/transpose_1/permÇ
9model_13/fixed_adjacency_graph_convolution_13/transpose_1	TransposePmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp:value:0Gmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2;
9model_13/fixed_adjacency_graph_convolution_13/transpose_1Ï
=model_13/fixed_adjacency_graph_convolution_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2?
=model_13/fixed_adjacency_graph_convolution_13/Reshape_1/shape­
7model_13/fixed_adjacency_graph_convolution_13/Reshape_1Reshape=model_13/fixed_adjacency_graph_convolution_13/transpose_1:y:0Fmodel_13/fixed_adjacency_graph_convolution_13/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF29
7model_13/fixed_adjacency_graph_convolution_13/Reshape_1ª
4model_13/fixed_adjacency_graph_convolution_13/MatMulMatMul>model_13/fixed_adjacency_graph_convolution_13/Reshape:output:0@model_13/fixed_adjacency_graph_convolution_13/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF26
4model_13/fixed_adjacency_graph_convolution_13/MatMulÄ
?model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/1Ä
?model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2A
?model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/2
=model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shapePack>model_13/fixed_adjacency_graph_convolution_13/unstack:output:0Hmodel_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/1:output:0Hmodel_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2?
=model_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape»
7model_13/fixed_adjacency_graph_convolution_13/Reshape_2Reshape>model_13/fixed_adjacency_graph_convolution_13/MatMul:product:0Fmodel_13/fixed_adjacency_graph_convolution_13/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF29
7model_13/fixed_adjacency_graph_convolution_13/Reshape_2Õ
>model_13/fixed_adjacency_graph_convolution_13/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2@
>model_13/fixed_adjacency_graph_convolution_13/transpose_2/permÄ
9model_13/fixed_adjacency_graph_convolution_13/transpose_2	Transpose@model_13/fixed_adjacency_graph_convolution_13/Reshape_2:output:0Gmodel_13/fixed_adjacency_graph_convolution_13/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2;
9model_13/fixed_adjacency_graph_convolution_13/transpose_2Û
5model_13/fixed_adjacency_graph_convolution_13/Shape_2Shape=model_13/fixed_adjacency_graph_convolution_13/transpose_2:y:0*
T0*
_output_shapes
:27
5model_13/fixed_adjacency_graph_convolution_13/Shape_2ì
7model_13/fixed_adjacency_graph_convolution_13/unstack_2Unpack>model_13/fixed_adjacency_graph_convolution_13/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num29
7model_13/fixed_adjacency_graph_convolution_13/unstack_2
Dmodel_13/fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOpReadVariableOpMmodel_13_fixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02F
Dmodel_13/fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOp¿
5model_13/fixed_adjacency_graph_convolution_13/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      27
5model_13/fixed_adjacency_graph_convolution_13/Shape_3ê
7model_13/fixed_adjacency_graph_convolution_13/unstack_3Unpack>model_13/fixed_adjacency_graph_convolution_13/Shape_3:output:0*
T0*
_output_shapes
: : *	
num29
7model_13/fixed_adjacency_graph_convolution_13/unstack_3Ï
=model_13/fixed_adjacency_graph_convolution_13/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=model_13/fixed_adjacency_graph_convolution_13/Reshape_3/shape¶
7model_13/fixed_adjacency_graph_convolution_13/Reshape_3Reshape=model_13/fixed_adjacency_graph_convolution_13/transpose_2:y:0Fmodel_13/fixed_adjacency_graph_convolution_13/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7model_13/fixed_adjacency_graph_convolution_13/Reshape_3¢
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOpReadVariableOpMmodel_13_fixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02J
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOpÑ
>model_13/fixed_adjacency_graph_convolution_13/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2@
>model_13/fixed_adjacency_graph_convolution_13/transpose_3/permÇ
9model_13/fixed_adjacency_graph_convolution_13/transpose_3	TransposePmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp:value:0Gmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/perm:output:0*
T0*
_output_shapes

:2;
9model_13/fixed_adjacency_graph_convolution_13/transpose_3Ï
=model_13/fixed_adjacency_graph_convolution_13/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2?
=model_13/fixed_adjacency_graph_convolution_13/Reshape_4/shape­
7model_13/fixed_adjacency_graph_convolution_13/Reshape_4Reshape=model_13/fixed_adjacency_graph_convolution_13/transpose_3:y:0Fmodel_13/fixed_adjacency_graph_convolution_13/Reshape_4/shape:output:0*
T0*
_output_shapes

:29
7model_13/fixed_adjacency_graph_convolution_13/Reshape_4°
6model_13/fixed_adjacency_graph_convolution_13/MatMul_1MatMul@model_13/fixed_adjacency_graph_convolution_13/Reshape_3:output:0@model_13/fixed_adjacency_graph_convolution_13/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6model_13/fixed_adjacency_graph_convolution_13/MatMul_1Ä
?model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2A
?model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/1Ä
?model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/2
=model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shapePack@model_13/fixed_adjacency_graph_convolution_13/unstack_2:output:0Hmodel_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/1:output:0Hmodel_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2?
=model_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape½
7model_13/fixed_adjacency_graph_convolution_13/Reshape_5Reshape@model_13/fixed_adjacency_graph_convolution_13/MatMul_1:product:0Fmodel_13/fixed_adjacency_graph_convolution_13/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF29
7model_13/fixed_adjacency_graph_convolution_13/Reshape_5
@model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOpReadVariableOpImodel_13_fixed_adjacency_graph_convolution_13_add_readvariableop_resource*
_output_shapes

:F*
dtype02B
@model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOp±
1model_13/fixed_adjacency_graph_convolution_13/addAddV2@model_13/fixed_adjacency_graph_convolution_13/Reshape_5:output:0Hmodel_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF23
1model_13/fixed_adjacency_graph_convolution_13/add
model_13/reshape_40/ShapeShape5model_13/fixed_adjacency_graph_convolution_13/add:z:0*
T0*
_output_shapes
:2
model_13/reshape_40/Shape
'model_13/reshape_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_13/reshape_40/strided_slice/stack 
)model_13/reshape_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_40/strided_slice/stack_1 
)model_13/reshape_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_40/strided_slice/stack_2Ú
!model_13/reshape_40/strided_sliceStridedSlice"model_13/reshape_40/Shape:output:00model_13/reshape_40/strided_slice/stack:output:02model_13/reshape_40/strided_slice/stack_1:output:02model_13/reshape_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_13/reshape_40/strided_slice
#model_13/reshape_40/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_13/reshape_40/Reshape/shape/1
#model_13/reshape_40/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model_13/reshape_40/Reshape/shape/2
#model_13/reshape_40/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_40/Reshape/shape/3²
!model_13/reshape_40/Reshape/shapePack*model_13/reshape_40/strided_slice:output:0,model_13/reshape_40/Reshape/shape/1:output:0,model_13/reshape_40/Reshape/shape/2:output:0,model_13/reshape_40/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!model_13/reshape_40/Reshape/shapeâ
model_13/reshape_40/ReshapeReshape5model_13/fixed_adjacency_graph_convolution_13/add:z:0*model_13/reshape_40/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/reshape_40/Reshape¡
"model_13/permute_13/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"model_13/permute_13/transpose/permØ
model_13/permute_13/transpose	Transpose$model_13/reshape_40/Reshape:output:0+model_13/permute_13/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/permute_13/transpose
model_13/reshape_41/ShapeShape!model_13/permute_13/transpose:y:0*
T0*
_output_shapes
:2
model_13/reshape_41/Shape
'model_13/reshape_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_13/reshape_41/strided_slice/stack 
)model_13/reshape_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_41/strided_slice/stack_1 
)model_13/reshape_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_41/strided_slice/stack_2Ú
!model_13/reshape_41/strided_sliceStridedSlice"model_13/reshape_41/Shape:output:00model_13/reshape_41/strided_slice/stack:output:02model_13/reshape_41/strided_slice/stack_1:output:02model_13/reshape_41/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_13/reshape_41/strided_slice
#model_13/reshape_41/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model_13/reshape_41/Reshape/shape/1
#model_13/reshape_41/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_13/reshape_41/Reshape/shape/2
!model_13/reshape_41/Reshape/shapePack*model_13/reshape_41/strided_slice:output:0,model_13/reshape_41/Reshape/shape/1:output:0,model_13/reshape_41/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_13/reshape_41/Reshape/shapeÊ
model_13/reshape_41/ReshapeReshape!model_13/permute_13/transpose:y:0*model_13/reshape_41/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/reshape_41/Reshape
model_13/lstm_13/ShapeShape$model_13/reshape_41/Reshape:output:0*
T0*
_output_shapes
:2
model_13/lstm_13/Shape
$model_13/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_13/lstm_13/strided_slice/stack
&model_13/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_13/lstm_13/strided_slice/stack_1
&model_13/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_13/lstm_13/strided_slice/stack_2È
model_13/lstm_13/strided_sliceStridedSlicemodel_13/lstm_13/Shape:output:0-model_13/lstm_13/strided_slice/stack:output:0/model_13/lstm_13/strided_slice/stack_1:output:0/model_13/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_13/lstm_13/strided_slice~
model_13/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
model_13/lstm_13/zeros/mul/y°
model_13/lstm_13/zeros/mulMul'model_13/lstm_13/strided_slice:output:0%model_13/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/zeros/mul
model_13/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
model_13/lstm_13/zeros/Less/y«
model_13/lstm_13/zeros/LessLessmodel_13/lstm_13/zeros/mul:z:0&model_13/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/zeros/Less
model_13/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2!
model_13/lstm_13/zeros/packed/1Ç
model_13/lstm_13/zeros/packedPack'model_13/lstm_13/strided_slice:output:0(model_13/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_13/lstm_13/zeros/packed
model_13/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_13/lstm_13/zeros/Const¹
model_13/lstm_13/zerosFill&model_13/lstm_13/zeros/packed:output:0%model_13/lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model_13/lstm_13/zeros
model_13/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2 
model_13/lstm_13/zeros_1/mul/y¶
model_13/lstm_13/zeros_1/mulMul'model_13/lstm_13/strided_slice:output:0'model_13/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/zeros_1/mul
model_13/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2!
model_13/lstm_13/zeros_1/Less/y³
model_13/lstm_13/zeros_1/LessLess model_13/lstm_13/zeros_1/mul:z:0(model_13/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_13/lstm_13/zeros_1/Less
!model_13/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2#
!model_13/lstm_13/zeros_1/packed/1Í
model_13/lstm_13/zeros_1/packedPack'model_13/lstm_13/strided_slice:output:0*model_13/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
model_13/lstm_13/zeros_1/packed
model_13/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
model_13/lstm_13/zeros_1/ConstÁ
model_13/lstm_13/zeros_1Fill(model_13/lstm_13/zeros_1/packed:output:0'model_13/lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model_13/lstm_13/zeros_1
model_13/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_13/lstm_13/transpose/permË
model_13/lstm_13/transpose	Transpose$model_13/reshape_41/Reshape:output:0(model_13/lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/lstm_13/transpose
model_13/lstm_13/Shape_1Shapemodel_13/lstm_13/transpose:y:0*
T0*
_output_shapes
:2
model_13/lstm_13/Shape_1
&model_13/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_13/lstm_13/strided_slice_1/stack
(model_13/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_13/lstm_13/strided_slice_1/stack_1
(model_13/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_13/lstm_13/strided_slice_1/stack_2Ô
 model_13/lstm_13/strided_slice_1StridedSlice!model_13/lstm_13/Shape_1:output:0/model_13/lstm_13/strided_slice_1/stack:output:01model_13/lstm_13/strided_slice_1/stack_1:output:01model_13/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_13/lstm_13/strided_slice_1§
,model_13/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,model_13/lstm_13/TensorArrayV2/element_shapeö
model_13/lstm_13/TensorArrayV2TensorListReserve5model_13/lstm_13/TensorArrayV2/element_shape:output:0)model_13/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_13/lstm_13/TensorArrayV2á
Fmodel_13/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2H
Fmodel_13/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape¼
8model_13/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_13/lstm_13/transpose:y:0Omodel_13/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8model_13/lstm_13/TensorArrayUnstack/TensorListFromTensor
&model_13/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_13/lstm_13/strided_slice_2/stack
(model_13/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_13/lstm_13/strided_slice_2/stack_1
(model_13/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_13/lstm_13/strided_slice_2/stack_2â
 model_13/lstm_13/strided_slice_2StridedSlicemodel_13/lstm_13/transpose:y:0/model_13/lstm_13/strided_slice_2/stack:output:01model_13/lstm_13/strided_slice_2/stack_1:output:01model_13/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2"
 model_13/lstm_13/strided_slice_2è
3model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp<model_13_lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype025
3model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOpñ
$model_13/lstm_13/lstm_cell_13/MatMulMatMul)model_13/lstm_13/strided_slice_2:output:0;model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_13/lstm_13/lstm_cell_13/MatMulî
5model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp>model_13_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype027
5model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpí
&model_13/lstm_13/lstm_cell_13/MatMul_1MatMulmodel_13/lstm_13/zeros:output:0=model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_13/lstm_13/lstm_cell_13/MatMul_1ä
!model_13/lstm_13/lstm_cell_13/addAddV2.model_13/lstm_13/lstm_cell_13/MatMul:product:00model_13/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model_13/lstm_13/lstm_cell_13/addç
4model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp=model_13_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpñ
%model_13/lstm_13/lstm_cell_13/BiasAddBiasAdd%model_13/lstm_13/lstm_cell_13/add:z:0<model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_13/lstm_13/lstm_cell_13/BiasAdd
#model_13/lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/lstm_13/lstm_cell_13/Const 
-model_13/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-model_13/lstm_13/lstm_cell_13/split/split_dim·
#model_13/lstm_13/lstm_cell_13/splitSplit6model_13/lstm_13/lstm_cell_13/split/split_dim:output:0.model_13/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2%
#model_13/lstm_13/lstm_cell_13/split¹
%model_13/lstm_13/lstm_cell_13/SigmoidSigmoid,model_13/lstm_13/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2'
%model_13/lstm_13/lstm_cell_13/Sigmoid½
'model_13/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid,model_13/lstm_13/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'model_13/lstm_13/lstm_cell_13/Sigmoid_1Ï
!model_13/lstm_13/lstm_cell_13/mulMul+model_13/lstm_13/lstm_cell_13/Sigmoid_1:y:0!model_13/lstm_13/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!model_13/lstm_13/lstm_cell_13/mul°
"model_13/lstm_13/lstm_cell_13/ReluRelu,model_13/lstm_13/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"model_13/lstm_13/lstm_cell_13/Reluà
#model_13/lstm_13/lstm_cell_13/mul_1Mul)model_13/lstm_13/lstm_cell_13/Sigmoid:y:00model_13/lstm_13/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2%
#model_13/lstm_13/lstm_cell_13/mul_1Õ
#model_13/lstm_13/lstm_cell_13/add_1AddV2%model_13/lstm_13/lstm_cell_13/mul:z:0'model_13/lstm_13/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2%
#model_13/lstm_13/lstm_cell_13/add_1½
'model_13/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid,model_13/lstm_13/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'model_13/lstm_13/lstm_cell_13/Sigmoid_2¯
$model_13/lstm_13/lstm_cell_13/Relu_1Relu'model_13/lstm_13/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2&
$model_13/lstm_13/lstm_cell_13/Relu_1ä
#model_13/lstm_13/lstm_cell_13/mul_2Mul+model_13/lstm_13/lstm_cell_13/Sigmoid_2:y:02model_13/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2%
#model_13/lstm_13/lstm_cell_13/mul_2±
.model_13/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   20
.model_13/lstm_13/TensorArrayV2_1/element_shapeü
 model_13/lstm_13/TensorArrayV2_1TensorListReserve7model_13/lstm_13/TensorArrayV2_1/element_shape:output:0)model_13/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 model_13/lstm_13/TensorArrayV2_1p
model_13/lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_13/lstm_13/time¡
)model_13/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)model_13/lstm_13/while/maximum_iterations
#model_13/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model_13/lstm_13/while/loop_counterí
model_13/lstm_13/whileWhile,model_13/lstm_13/while/loop_counter:output:02model_13/lstm_13/while/maximum_iterations:output:0model_13/lstm_13/time:output:0)model_13/lstm_13/TensorArrayV2_1:handle:0model_13/lstm_13/zeros:output:0!model_13/lstm_13/zeros_1:output:0)model_13/lstm_13/strided_slice_1:output:0Hmodel_13/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0<model_13_lstm_13_lstm_cell_13_matmul_readvariableop_resource>model_13_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource=model_13_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"model_13_lstm_13_while_body_234043*.
cond&R$
"model_13_lstm_13_while_cond_234042*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
model_13/lstm_13/while×
Amodel_13/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2C
Amodel_13/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape¬
3model_13/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStackmodel_13/lstm_13/while:output:3Jmodel_13/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype025
3model_13/lstm_13/TensorArrayV2Stack/TensorListStack£
&model_13/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2(
&model_13/lstm_13/strided_slice_3/stack
(model_13/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(model_13/lstm_13/strided_slice_3/stack_1
(model_13/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_13/lstm_13/strided_slice_3/stack_2
 model_13/lstm_13/strided_slice_3StridedSlice<model_13/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0/model_13/lstm_13/strided_slice_3/stack:output:01model_13/lstm_13/strided_slice_3/stack_1:output:01model_13/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2"
 model_13/lstm_13/strided_slice_3
!model_13/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!model_13/lstm_13/transpose_1/permé
model_13/lstm_13/transpose_1	Transpose<model_13/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0*model_13/lstm_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model_13/lstm_13/transpose_1
model_13/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_13/lstm_13/runtime¥
model_13/dropout_13/IdentityIdentity)model_13/lstm_13/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model_13/dropout_13/IdentityÃ
'model_13/dense_13/MatMul/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02)
'model_13/dense_13/MatMul/ReadVariableOpÈ
model_13/dense_13/MatMulMatMul%model_13/dropout_13/Identity:output:0/model_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/dense_13/MatMulÂ
(model_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02*
(model_13/dense_13/BiasAdd/ReadVariableOpÉ
model_13/dense_13/BiasAddBiasAdd"model_13/dense_13/MatMul:product:00model_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/dense_13/BiasAdd
model_13/dense_13/SigmoidSigmoid"model_13/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
model_13/dense_13/SigmoidÝ
IdentityIdentitymodel_13/dense_13/Sigmoid:y:0)^model_13/dense_13/BiasAdd/ReadVariableOp(^model_13/dense_13/MatMul/ReadVariableOpA^model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOpI^model_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpI^model_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp5^model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp4^model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOp6^model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^model_13/lstm_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2T
(model_13/dense_13/BiasAdd/ReadVariableOp(model_13/dense_13/BiasAdd/ReadVariableOp2R
'model_13/dense_13/MatMul/ReadVariableOp'model_13/dense_13/MatMul/ReadVariableOp2
@model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOp@model_13/fixed_adjacency_graph_convolution_13/add/ReadVariableOp2
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpHmodel_13/fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp2
Hmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOpHmodel_13/fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp2l
4model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp4model_13/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2j
3model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOp3model_13/lstm_13/lstm_cell_13/MatMul/ReadVariableOp2n
5model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp5model_13/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp20
model_13/lstm_13/whilemodel_13/lstm_13/while:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
input_27
[
ò
C__inference_lstm_13_layer_call_and_return_conditional_losses_235208

inputs/
+lstm_cell_13_matmul_readvariableop_resource1
-lstm_cell_13_matmul_1_readvariableop_resource0
,lstm_cell_13_biasadd_readvariableop_resource
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul»
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAddj
lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/Const~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimó
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul}
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Sigmoid_2|
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/Relu_1 
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterî
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_235123*
condR
while_cond_235122*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeæ
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Çù
á
D__inference_model_13_layer_call_and_return_conditional_losses_235709

inputsH
Dfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resourceH
Dfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resourceD
@fixed_adjacency_graph_convolution_13_add_readvariableop_resource7
3lstm_13_lstm_cell_13_matmul_readvariableop_resource9
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource8
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢7fixed_adjacency_graph_convolution_13/add/ReadVariableOp¢?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp¢?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/while
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim·
tf.expand_dims_13/ExpandDims
ExpandDimsinputs)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsy
reshape_39/ShapeShape%tf.expand_dims_13/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_39/Shape
reshape_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_39/strided_slice/stack
 reshape_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_39/strided_slice/stack_1
 reshape_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_39/strided_slice/stack_2¤
reshape_39/strided_sliceStridedSlicereshape_39/Shape:output:0'reshape_39/strided_slice/stack:output:0)reshape_39/strided_slice/stack_1:output:0)reshape_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_39/strided_slicez
reshape_39/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_39/Reshape/shape/1z
reshape_39/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_39/Reshape/shape/2×
reshape_39/Reshape/shapePack!reshape_39/strided_slice:output:0#reshape_39/Reshape/shape/1:output:0#reshape_39/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_39/Reshape/shape³
reshape_39/ReshapeReshape%tf.expand_dims_13/ExpandDims:output:0!reshape_39/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_39/Reshape¿
3fixed_adjacency_graph_convolution_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          25
3fixed_adjacency_graph_convolution_13/transpose/permþ
.fixed_adjacency_graph_convolution_13/transpose	Transposereshape_39/Reshape:output:0<fixed_adjacency_graph_convolution_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/transposeº
*fixed_adjacency_graph_convolution_13/ShapeShape2fixed_adjacency_graph_convolution_13/transpose:y:0*
T0*
_output_shapes
:2,
*fixed_adjacency_graph_convolution_13/ShapeË
,fixed_adjacency_graph_convolution_13/unstackUnpack3fixed_adjacency_graph_convolution_13/Shape:output:0*
T0*
_output_shapes
: : : *	
num2.
,fixed_adjacency_graph_convolution_13/unstackÿ
;fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02=
;fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOp­
,fixed_adjacency_graph_convolution_13/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2.
,fixed_adjacency_graph_convolution_13/Shape_1Ï
.fixed_adjacency_graph_convolution_13/unstack_1Unpack5fixed_adjacency_graph_convolution_13/Shape_1:output:0*
T0*
_output_shapes
: : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_1¹
2fixed_adjacency_graph_convolution_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   24
2fixed_adjacency_graph_convolution_13/Reshape/shape
,fixed_adjacency_graph_convolution_13/ReshapeReshape2fixed_adjacency_graph_convolution_13/transpose:y:0;fixed_adjacency_graph_convolution_13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2.
,fixed_adjacency_graph_convolution_13/Reshape
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02A
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp¿
5fixed_adjacency_graph_convolution_13/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5fixed_adjacency_graph_convolution_13/transpose_1/perm£
0fixed_adjacency_graph_convolution_13/transpose_1	TransposeGfixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp:value:0>fixed_adjacency_graph_convolution_13/transpose_1/perm:output:0*
T0*
_output_shapes

:FF22
0fixed_adjacency_graph_convolution_13/transpose_1½
4fixed_adjacency_graph_convolution_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ26
4fixed_adjacency_graph_convolution_13/Reshape_1/shape
.fixed_adjacency_graph_convolution_13/Reshape_1Reshape4fixed_adjacency_graph_convolution_13/transpose_1:y:0=fixed_adjacency_graph_convolution_13/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF20
.fixed_adjacency_graph_convolution_13/Reshape_1
+fixed_adjacency_graph_convolution_13/MatMulMatMul5fixed_adjacency_graph_convolution_13/Reshape:output:07fixed_adjacency_graph_convolution_13/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2-
+fixed_adjacency_graph_convolution_13/MatMul²
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/1²
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F28
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/2Û
4fixed_adjacency_graph_convolution_13/Reshape_2/shapePack5fixed_adjacency_graph_convolution_13/unstack:output:0?fixed_adjacency_graph_convolution_13/Reshape_2/shape/1:output:0?fixed_adjacency_graph_convolution_13/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:26
4fixed_adjacency_graph_convolution_13/Reshape_2/shape
.fixed_adjacency_graph_convolution_13/Reshape_2Reshape5fixed_adjacency_graph_convolution_13/MatMul:product:0=fixed_adjacency_graph_convolution_13/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/Reshape_2Ã
5fixed_adjacency_graph_convolution_13/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          27
5fixed_adjacency_graph_convolution_13/transpose_2/perm 
0fixed_adjacency_graph_convolution_13/transpose_2	Transpose7fixed_adjacency_graph_convolution_13/Reshape_2:output:0>fixed_adjacency_graph_convolution_13/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF22
0fixed_adjacency_graph_convolution_13/transpose_2À
,fixed_adjacency_graph_convolution_13/Shape_2Shape4fixed_adjacency_graph_convolution_13/transpose_2:y:0*
T0*
_output_shapes
:2.
,fixed_adjacency_graph_convolution_13/Shape_2Ñ
.fixed_adjacency_graph_convolution_13/unstack_2Unpack5fixed_adjacency_graph_convolution_13/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_2ÿ
;fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02=
;fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOp­
,fixed_adjacency_graph_convolution_13/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2.
,fixed_adjacency_graph_convolution_13/Shape_3Ï
.fixed_adjacency_graph_convolution_13/unstack_3Unpack5fixed_adjacency_graph_convolution_13/Shape_3:output:0*
T0*
_output_shapes
: : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_3½
4fixed_adjacency_graph_convolution_13/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   26
4fixed_adjacency_graph_convolution_13/Reshape_3/shape
.fixed_adjacency_graph_convolution_13/Reshape_3Reshape4fixed_adjacency_graph_convolution_13/transpose_2:y:0=fixed_adjacency_graph_convolution_13/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.fixed_adjacency_graph_convolution_13/Reshape_3
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02A
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp¿
5fixed_adjacency_graph_convolution_13/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5fixed_adjacency_graph_convolution_13/transpose_3/perm£
0fixed_adjacency_graph_convolution_13/transpose_3	TransposeGfixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp:value:0>fixed_adjacency_graph_convolution_13/transpose_3/perm:output:0*
T0*
_output_shapes

:22
0fixed_adjacency_graph_convolution_13/transpose_3½
4fixed_adjacency_graph_convolution_13/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ26
4fixed_adjacency_graph_convolution_13/Reshape_4/shape
.fixed_adjacency_graph_convolution_13/Reshape_4Reshape4fixed_adjacency_graph_convolution_13/transpose_3:y:0=fixed_adjacency_graph_convolution_13/Reshape_4/shape:output:0*
T0*
_output_shapes

:20
.fixed_adjacency_graph_convolution_13/Reshape_4
-fixed_adjacency_graph_convolution_13/MatMul_1MatMul7fixed_adjacency_graph_convolution_13/Reshape_3:output:07fixed_adjacency_graph_convolution_13/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-fixed_adjacency_graph_convolution_13/MatMul_1²
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F28
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/1²
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :28
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/2Ý
4fixed_adjacency_graph_convolution_13/Reshape_5/shapePack7fixed_adjacency_graph_convolution_13/unstack_2:output:0?fixed_adjacency_graph_convolution_13/Reshape_5/shape/1:output:0?fixed_adjacency_graph_convolution_13/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:26
4fixed_adjacency_graph_convolution_13/Reshape_5/shape
.fixed_adjacency_graph_convolution_13/Reshape_5Reshape7fixed_adjacency_graph_convolution_13/MatMul_1:product:0=fixed_adjacency_graph_convolution_13/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/Reshape_5ó
7fixed_adjacency_graph_convolution_13/add/ReadVariableOpReadVariableOp@fixed_adjacency_graph_convolution_13_add_readvariableop_resource*
_output_shapes

:F*
dtype029
7fixed_adjacency_graph_convolution_13/add/ReadVariableOp
(fixed_adjacency_graph_convolution_13/addAddV27fixed_adjacency_graph_convolution_13/Reshape_5:output:0?fixed_adjacency_graph_convolution_13/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2*
(fixed_adjacency_graph_convolution_13/add
reshape_40/ShapeShape,fixed_adjacency_graph_convolution_13/add:z:0*
T0*
_output_shapes
:2
reshape_40/Shape
reshape_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_40/strided_slice/stack
 reshape_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_40/strided_slice/stack_1
 reshape_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_40/strided_slice/stack_2¤
reshape_40/strided_sliceStridedSlicereshape_40/Shape:output:0'reshape_40/strided_slice/stack:output:0)reshape_40/strided_slice/stack_1:output:0)reshape_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_40/strided_slicez
reshape_40/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_40/Reshape/shape/1
reshape_40/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_40/Reshape/shape/2z
reshape_40/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_40/Reshape/shape/3ü
reshape_40/Reshape/shapePack!reshape_40/strided_slice:output:0#reshape_40/Reshape/shape/1:output:0#reshape_40/Reshape/shape/2:output:0#reshape_40/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_40/Reshape/shape¾
reshape_40/ReshapeReshape,fixed_adjacency_graph_convolution_13/add:z:0!reshape_40/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_40/Reshape
permute_13/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_13/transpose/perm´
permute_13/transpose	Transposereshape_40/Reshape:output:0"permute_13/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
permute_13/transposel
reshape_41/ShapeShapepermute_13/transpose:y:0*
T0*
_output_shapes
:2
reshape_41/Shape
reshape_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_41/strided_slice/stack
 reshape_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_41/strided_slice/stack_1
 reshape_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_41/strided_slice/stack_2¤
reshape_41/strided_sliceStridedSlicereshape_41/Shape:output:0'reshape_41/strided_slice/stack:output:0)reshape_41/strided_slice/stack_1:output:0)reshape_41/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_41/strided_slice
reshape_41/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_41/Reshape/shape/1z
reshape_41/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_41/Reshape/shape/2×
reshape_41/Reshape/shapePack!reshape_41/strided_slice:output:0#reshape_41/Reshape/shape/1:output:0#reshape_41/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_41/Reshape/shape¦
reshape_41/ReshapeReshapepermute_13/transpose:y:0!reshape_41/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_41/Reshapei
lstm_13/ShapeShapereshape_41/Reshape:output:0*
T0*
_output_shapes
:2
lstm_13/Shape
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicel
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros/mul/y
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_13/zeros/Less/y
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lessr
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros/packed/1£
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/zerosp
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros_1/mul/y
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_13/zeros_1/Less/y
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessv
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros_1/packed/1©
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/zeros_1
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm§
lstm_13/transpose	Transposereshape_41/Reshape:output:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/TensorArrayV2/element_shapeÒ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2Ï
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2¬
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
lstm_13/strided_slice_2Í
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpÍ
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMulÓ
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpÉ
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMul_1À
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/addÌ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpÍ
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/BiasAddz
lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/lstm_cell_13/Const
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_13/lstm_cell_13/split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Sigmoid¢
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/lstm_cell_13/Sigmoid_1«
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Relu¼
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul_1±
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/add_1¢
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/lstm_cell_13/Sigmoid_2
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Relu_1À
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul_2
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2'
%lstm_13/TensorArrayV2_1/element_shapeØ
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counteræ
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_13_while_body_235609*%
condR
lstm_13_while_cond_235608*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
lstm_13/whileÅ
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_13/strided_slice_3/stack
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2Ê
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
lstm_13/strided_slice_3
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/permÅ
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtimey
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_13/dropout/Const®
dropout_13/dropout/MulMul lstm_13/strided_slice_3:output:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShape lstm_13/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_13/dropout/ShapeÕ
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_13/dropout/GreaterEqual/yê
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
dropout_13/dropout/GreaterEqual 
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_13/dropout/Cast¦
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_13/dropout/Mul_1¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_13/MatMul/ReadVariableOp¤
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/BiasAdd|
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/Sigmoid
IdentityIdentitydense_13/Sigmoid:y:0 ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp8^fixed_adjacency_graph_convolution_13/add/ReadVariableOp@^fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp@^fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2r
7fixed_adjacency_graph_convolution_13/add/ReadVariableOp7fixed_adjacency_graph_convolution_13/add/ReadVariableOp2
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp2
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs


ã
lstm_13_while_cond_235857,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1D
@lstm_13_while_lstm_13_while_cond_235857___redundant_placeholder0D
@lstm_13_while_lstm_13_while_cond_235857___redundant_placeholder1D
@lstm_13_while_lstm_13_while_cond_235857___redundant_placeholder2D
@lstm_13_while_lstm_13_while_cond_235857___redundant_placeholder3
lstm_13_while_identity

lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
®
G
+__inference_reshape_39_layer_call_fn_236011

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_39_layer_call_and_return_conditional_losses_2347782
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs

·
"__inference__traced_restore_237134
file_prefix;
7assignvariableop_fixed_adjacency_graph_convolution_13_aB
>assignvariableop_1_fixed_adjacency_graph_convolution_13_kernel@
<assignvariableop_2_fixed_adjacency_graph_convolution_13_bias&
"assignvariableop_3_dense_13_kernel$
 assignvariableop_4_dense_13_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate3
/assignvariableop_10_lstm_13_lstm_cell_13_kernel=
9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernel1
-assignvariableop_12_lstm_13_lstm_cell_13_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1J
Fassignvariableop_17_adam_fixed_adjacency_graph_convolution_13_kernel_mH
Dassignvariableop_18_adam_fixed_adjacency_graph_convolution_13_bias_m.
*assignvariableop_19_adam_dense_13_kernel_m,
(assignvariableop_20_adam_dense_13_bias_m:
6assignvariableop_21_adam_lstm_13_lstm_cell_13_kernel_mD
@assignvariableop_22_adam_lstm_13_lstm_cell_13_recurrent_kernel_m8
4assignvariableop_23_adam_lstm_13_lstm_cell_13_bias_mJ
Fassignvariableop_24_adam_fixed_adjacency_graph_convolution_13_kernel_vH
Dassignvariableop_25_adam_fixed_adjacency_graph_convolution_13_bias_v.
*assignvariableop_26_adam_dense_13_kernel_v,
(assignvariableop_27_adam_dense_13_bias_v:
6assignvariableop_28_adam_lstm_13_lstm_cell_13_kernel_vD
@assignvariableop_29_adam_lstm_13_lstm_cell_13_recurrent_kernel_v8
4assignvariableop_30_adam_lstm_13_lstm_cell_13_bias_v
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9õ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value÷Bô B1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¶
AssignVariableOpAssignVariableOp7assignvariableop_fixed_adjacency_graph_convolution_13_aIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ã
AssignVariableOp_1AssignVariableOp>assignvariableop_1_fixed_adjacency_graph_convolution_13_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Á
AssignVariableOp_2AssignVariableOp<assignvariableop_2_fixed_adjacency_graph_convolution_13_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_13_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_13_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5¡
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¢
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ª
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_13_lstm_cell_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12µ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_13_lstm_cell_13_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Î
AssignVariableOp_17AssignVariableOpFassignvariableop_17_adam_fixed_adjacency_graph_convolution_13_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ì
AssignVariableOp_18AssignVariableOpDassignvariableop_18_adam_fixed_adjacency_graph_convolution_13_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_13_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_13_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¾
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_lstm_13_lstm_cell_13_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22È
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¼
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_13_lstm_cell_13_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Î
AssignVariableOp_24AssignVariableOpFassignvariableop_24_adam_fixed_adjacency_graph_convolution_13_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ì
AssignVariableOp_25AssignVariableOpDassignvariableop_25_adam_fixed_adjacency_graph_convolution_13_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_13_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27°
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_13_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¾
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_13_lstm_cell_13_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29È
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¼
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_13_lstm_cell_13_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31û
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*
_input_shapes
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
«
Ã
while_cond_234680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234680___redundant_placeholder04
0while_while_cond_234680___redundant_placeholder14
0while_while_cond_234680___redundant_placeholder24
0while_while_cond_234680___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
«
Û
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_234255

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
mul_2¨
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity¬

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1¬

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿF:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
ð
á
D__inference_model_13_layer_call_and_return_conditional_losses_235951

inputsH
Dfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resourceH
Dfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resourceD
@fixed_adjacency_graph_convolution_13_add_readvariableop_resource7
3lstm_13_lstm_cell_13_matmul_readvariableop_resource9
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource8
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢7fixed_adjacency_graph_convolution_13/add/ReadVariableOp¢?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp¢?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/while
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim·
tf.expand_dims_13/ExpandDims
ExpandDimsinputs)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsy
reshape_39/ShapeShape%tf.expand_dims_13/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_39/Shape
reshape_39/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_39/strided_slice/stack
 reshape_39/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_39/strided_slice/stack_1
 reshape_39/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_39/strided_slice/stack_2¤
reshape_39/strided_sliceStridedSlicereshape_39/Shape:output:0'reshape_39/strided_slice/stack:output:0)reshape_39/strided_slice/stack_1:output:0)reshape_39/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_39/strided_slicez
reshape_39/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_39/Reshape/shape/1z
reshape_39/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_39/Reshape/shape/2×
reshape_39/Reshape/shapePack!reshape_39/strided_slice:output:0#reshape_39/Reshape/shape/1:output:0#reshape_39/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_39/Reshape/shape³
reshape_39/ReshapeReshape%tf.expand_dims_13/ExpandDims:output:0!reshape_39/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_39/Reshape¿
3fixed_adjacency_graph_convolution_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          25
3fixed_adjacency_graph_convolution_13/transpose/permþ
.fixed_adjacency_graph_convolution_13/transpose	Transposereshape_39/Reshape:output:0<fixed_adjacency_graph_convolution_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/transposeº
*fixed_adjacency_graph_convolution_13/ShapeShape2fixed_adjacency_graph_convolution_13/transpose:y:0*
T0*
_output_shapes
:2,
*fixed_adjacency_graph_convolution_13/ShapeË
,fixed_adjacency_graph_convolution_13/unstackUnpack3fixed_adjacency_graph_convolution_13/Shape:output:0*
T0*
_output_shapes
: : : *	
num2.
,fixed_adjacency_graph_convolution_13/unstackÿ
;fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02=
;fixed_adjacency_graph_convolution_13/Shape_1/ReadVariableOp­
,fixed_adjacency_graph_convolution_13/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2.
,fixed_adjacency_graph_convolution_13/Shape_1Ï
.fixed_adjacency_graph_convolution_13/unstack_1Unpack5fixed_adjacency_graph_convolution_13/Shape_1:output:0*
T0*
_output_shapes
: : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_1¹
2fixed_adjacency_graph_convolution_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   24
2fixed_adjacency_graph_convolution_13/Reshape/shape
,fixed_adjacency_graph_convolution_13/ReshapeReshape2fixed_adjacency_graph_convolution_13/transpose:y:0;fixed_adjacency_graph_convolution_13/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2.
,fixed_adjacency_graph_convolution_13/Reshape
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02A
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp¿
5fixed_adjacency_graph_convolution_13/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5fixed_adjacency_graph_convolution_13/transpose_1/perm£
0fixed_adjacency_graph_convolution_13/transpose_1	TransposeGfixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp:value:0>fixed_adjacency_graph_convolution_13/transpose_1/perm:output:0*
T0*
_output_shapes

:FF22
0fixed_adjacency_graph_convolution_13/transpose_1½
4fixed_adjacency_graph_convolution_13/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ26
4fixed_adjacency_graph_convolution_13/Reshape_1/shape
.fixed_adjacency_graph_convolution_13/Reshape_1Reshape4fixed_adjacency_graph_convolution_13/transpose_1:y:0=fixed_adjacency_graph_convolution_13/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF20
.fixed_adjacency_graph_convolution_13/Reshape_1
+fixed_adjacency_graph_convolution_13/MatMulMatMul5fixed_adjacency_graph_convolution_13/Reshape:output:07fixed_adjacency_graph_convolution_13/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2-
+fixed_adjacency_graph_convolution_13/MatMul²
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/1²
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F28
6fixed_adjacency_graph_convolution_13/Reshape_2/shape/2Û
4fixed_adjacency_graph_convolution_13/Reshape_2/shapePack5fixed_adjacency_graph_convolution_13/unstack:output:0?fixed_adjacency_graph_convolution_13/Reshape_2/shape/1:output:0?fixed_adjacency_graph_convolution_13/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:26
4fixed_adjacency_graph_convolution_13/Reshape_2/shape
.fixed_adjacency_graph_convolution_13/Reshape_2Reshape5fixed_adjacency_graph_convolution_13/MatMul:product:0=fixed_adjacency_graph_convolution_13/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/Reshape_2Ã
5fixed_adjacency_graph_convolution_13/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          27
5fixed_adjacency_graph_convolution_13/transpose_2/perm 
0fixed_adjacency_graph_convolution_13/transpose_2	Transpose7fixed_adjacency_graph_convolution_13/Reshape_2:output:0>fixed_adjacency_graph_convolution_13/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF22
0fixed_adjacency_graph_convolution_13/transpose_2À
,fixed_adjacency_graph_convolution_13/Shape_2Shape4fixed_adjacency_graph_convolution_13/transpose_2:y:0*
T0*
_output_shapes
:2.
,fixed_adjacency_graph_convolution_13/Shape_2Ñ
.fixed_adjacency_graph_convolution_13/unstack_2Unpack5fixed_adjacency_graph_convolution_13/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_2ÿ
;fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02=
;fixed_adjacency_graph_convolution_13/Shape_3/ReadVariableOp­
,fixed_adjacency_graph_convolution_13/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2.
,fixed_adjacency_graph_convolution_13/Shape_3Ï
.fixed_adjacency_graph_convolution_13/unstack_3Unpack5fixed_adjacency_graph_convolution_13/Shape_3:output:0*
T0*
_output_shapes
: : *	
num20
.fixed_adjacency_graph_convolution_13/unstack_3½
4fixed_adjacency_graph_convolution_13/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   26
4fixed_adjacency_graph_convolution_13/Reshape_3/shape
.fixed_adjacency_graph_convolution_13/Reshape_3Reshape4fixed_adjacency_graph_convolution_13/transpose_2:y:0=fixed_adjacency_graph_convolution_13/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.fixed_adjacency_graph_convolution_13/Reshape_3
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOpReadVariableOpDfixed_adjacency_graph_convolution_13_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02A
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp¿
5fixed_adjacency_graph_convolution_13/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5fixed_adjacency_graph_convolution_13/transpose_3/perm£
0fixed_adjacency_graph_convolution_13/transpose_3	TransposeGfixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp:value:0>fixed_adjacency_graph_convolution_13/transpose_3/perm:output:0*
T0*
_output_shapes

:22
0fixed_adjacency_graph_convolution_13/transpose_3½
4fixed_adjacency_graph_convolution_13/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ26
4fixed_adjacency_graph_convolution_13/Reshape_4/shape
.fixed_adjacency_graph_convolution_13/Reshape_4Reshape4fixed_adjacency_graph_convolution_13/transpose_3:y:0=fixed_adjacency_graph_convolution_13/Reshape_4/shape:output:0*
T0*
_output_shapes

:20
.fixed_adjacency_graph_convolution_13/Reshape_4
-fixed_adjacency_graph_convolution_13/MatMul_1MatMul7fixed_adjacency_graph_convolution_13/Reshape_3:output:07fixed_adjacency_graph_convolution_13/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-fixed_adjacency_graph_convolution_13/MatMul_1²
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F28
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/1²
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :28
6fixed_adjacency_graph_convolution_13/Reshape_5/shape/2Ý
4fixed_adjacency_graph_convolution_13/Reshape_5/shapePack7fixed_adjacency_graph_convolution_13/unstack_2:output:0?fixed_adjacency_graph_convolution_13/Reshape_5/shape/1:output:0?fixed_adjacency_graph_convolution_13/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:26
4fixed_adjacency_graph_convolution_13/Reshape_5/shape
.fixed_adjacency_graph_convolution_13/Reshape_5Reshape7fixed_adjacency_graph_convolution_13/MatMul_1:product:0=fixed_adjacency_graph_convolution_13/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF20
.fixed_adjacency_graph_convolution_13/Reshape_5ó
7fixed_adjacency_graph_convolution_13/add/ReadVariableOpReadVariableOp@fixed_adjacency_graph_convolution_13_add_readvariableop_resource*
_output_shapes

:F*
dtype029
7fixed_adjacency_graph_convolution_13/add/ReadVariableOp
(fixed_adjacency_graph_convolution_13/addAddV27fixed_adjacency_graph_convolution_13/Reshape_5:output:0?fixed_adjacency_graph_convolution_13/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2*
(fixed_adjacency_graph_convolution_13/add
reshape_40/ShapeShape,fixed_adjacency_graph_convolution_13/add:z:0*
T0*
_output_shapes
:2
reshape_40/Shape
reshape_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_40/strided_slice/stack
 reshape_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_40/strided_slice/stack_1
 reshape_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_40/strided_slice/stack_2¤
reshape_40/strided_sliceStridedSlicereshape_40/Shape:output:0'reshape_40/strided_slice/stack:output:0)reshape_40/strided_slice/stack_1:output:0)reshape_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_40/strided_slicez
reshape_40/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_40/Reshape/shape/1
reshape_40/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_40/Reshape/shape/2z
reshape_40/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_40/Reshape/shape/3ü
reshape_40/Reshape/shapePack!reshape_40/strided_slice:output:0#reshape_40/Reshape/shape/1:output:0#reshape_40/Reshape/shape/2:output:0#reshape_40/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_40/Reshape/shape¾
reshape_40/ReshapeReshape,fixed_adjacency_graph_convolution_13/add:z:0!reshape_40/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_40/Reshape
permute_13/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_13/transpose/perm´
permute_13/transpose	Transposereshape_40/Reshape:output:0"permute_13/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
permute_13/transposel
reshape_41/ShapeShapepermute_13/transpose:y:0*
T0*
_output_shapes
:2
reshape_41/Shape
reshape_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_41/strided_slice/stack
 reshape_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_41/strided_slice/stack_1
 reshape_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_41/strided_slice/stack_2¤
reshape_41/strided_sliceStridedSlicereshape_41/Shape:output:0'reshape_41/strided_slice/stack:output:0)reshape_41/strided_slice/stack_1:output:0)reshape_41/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_41/strided_slice
reshape_41/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
reshape_41/Reshape/shape/1z
reshape_41/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_41/Reshape/shape/2×
reshape_41/Reshape/shapePack!reshape_41/strided_slice:output:0#reshape_41/Reshape/shape/1:output:0#reshape_41/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_41/Reshape/shape¦
reshape_41/ReshapeReshapepermute_13/transpose:y:0!reshape_41/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
reshape_41/Reshapei
lstm_13/ShapeShapereshape_41/Reshape:output:0*
T0*
_output_shapes
:2
lstm_13/Shape
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicel
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros/mul/y
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_13/zeros/Less/y
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lessr
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros/packed/1£
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/zerosp
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros_1/mul/y
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_13/zeros_1/Less/y
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessv
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_13/zeros_1/packed/1©
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/zeros_1
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm§
lstm_13/transpose	Transposereshape_41/Reshape:output:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/TensorArrayV2/element_shapeÒ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2Ï
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2¬
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
lstm_13/strided_slice_2Í
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	F*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpÍ
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMulÓ
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpÉ
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMul_1À
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/addÌ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpÍ
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/BiasAddz
lstm_13/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/lstm_cell_13/Const
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
lstm_13/lstm_cell_13/split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Sigmoid¢
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/lstm_cell_13/Sigmoid_1«
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Relu¼
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul_1±
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/add_1¢
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
lstm_13/lstm_cell_13/Sigmoid_2
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/Relu_1À
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/lstm_cell_13/mul_2
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2'
%lstm_13/TensorArrayV2_1/element_shapeØ
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counteræ
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_13_while_body_235858*%
condR
lstm_13_while_cond_235857*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
lstm_13/whileÅ
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_13/strided_slice_3/stack
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2Ê
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
lstm_13/strided_slice_3
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/permÅ
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtime
dropout_13/IdentityIdentity lstm_13/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dropout_13/Identity¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_13/MatMul/ReadVariableOp¤
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/BiasAdd|
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
dense_13/Sigmoid
IdentityIdentitydense_13/Sigmoid:y:0 ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp8^fixed_adjacency_graph_convolution_13/add/ReadVariableOp@^fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp@^fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2r
7fixed_adjacency_graph_convolution_13/add/ReadVariableOp7fixed_adjacency_graph_convolution_13/add/ReadVariableOp2
?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp?fixed_adjacency_graph_convolution_13/transpose_1/ReadVariableOp2
?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp?fixed_adjacency_graph_convolution_13/transpose_3/ReadVariableOp2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ï
b
F__inference_reshape_39_layer_call_and_return_conditional_losses_236006

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
«
Ã
while_cond_236179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236179___redundant_placeholder04
0while_while_cond_236179___redundant_placeholder14
0while_while_cond_236179___redundant_placeholder24
0while_while_cond_236179___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
À,
»
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_236064
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity¢add/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
	Reshape_5
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
add®
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
"
_user_specified_name
features


(__inference_lstm_13_layer_call_fn_236429

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2350552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿF:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ü
~
)__inference_dense_13_layer_call_fn_236815

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2352792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
C
þ
while_body_236333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_13_matmul_readvariableop_resource_09
5while_lstm_cell_13_matmul_1_readvariableop_resource_08
4while_lstm_cell_13_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_13_matmul_readvariableop_resource7
3while_lstm_cell_13_matmul_1_readvariableop_resource6
2while_lstm_cell_13_biasadd_readvariableop_resource¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	F*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÏ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAddv
while/lstm_cell_13/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_13/Const
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_1 
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu´
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_1©
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/Relu_1¸
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/lstm_cell_13/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1â
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityõ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ä
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :::2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ÛD
Ü
C__inference_lstm_13_layer_call_and_return_conditional_losses_234618

inputs
lstm_cell_13_234536
lstm_cell_13_234538
lstm_cell_13_234540
identity¢$lstm_cell_13/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿF   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
shrink_axis_mask2
strided_slice_2
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_234536lstm_cell_13_234538lstm_cell_13_234540*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_2342222&
$lstm_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_234536lstm_cell_13_234538lstm_cell_13_234540*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_234549*
condR
while_cond_234548*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_13/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF:::2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
É
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_235255

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
º(

D__inference_model_13_layer_call_and_return_conditional_losses_235359

inputs/
+fixed_adjacency_graph_convolution_13_235335/
+fixed_adjacency_graph_convolution_13_235337/
+fixed_adjacency_graph_convolution_13_235339
lstm_13_235345
lstm_13_235347
lstm_13_235349
dense_13_235353
dense_13_235355
identity¢ dense_13/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 tf.expand_dims_13/ExpandDims/dim·
tf.expand_dims_13/ExpandDims
ExpandDimsinputs)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2
tf.expand_dims_13/ExpandDimsý
reshape_39/PartitionedCallPartitionedCall%tf.expand_dims_13/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_39_layer_call_and_return_conditional_losses_2347782
reshape_39/PartitionedCallð
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_39/PartitionedCall:output:0+fixed_adjacency_graph_convolution_13_235335+fixed_adjacency_graph_convolution_13_235337+fixed_adjacency_graph_convolution_13_235339*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_2348392>
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall¡
reshape_40/PartitionedCallPartitionedCallEfixed_adjacency_graph_convolution_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_40_layer_call_and_return_conditional_losses_2348732
reshape_40/PartitionedCallÿ
permute_13/PartitionedCallPartitionedCall#reshape_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_permute_13_layer_call_and_return_conditional_losses_2341432
permute_13/PartitionedCallû
reshape_41/PartitionedCallPartitionedCall#permute_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_41_layer_call_and_return_conditional_losses_2348952
reshape_41/PartitionedCall¾
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#reshape_41/PartitionedCall:output:0lstm_13_235345lstm_13_235347lstm_13_235349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_13_layer_call_and_return_conditional_losses_2350552!
lstm_13/StatefulPartitionedCall
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2352502$
"dropout_13/StatefulPartitionedCall¹
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_235353dense_13_235355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2352792"
 dense_13/StatefulPartitionedCall¦
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall=^fixed_adjacency_graph_convolution_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿF::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2|
<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall<fixed_adjacency_graph_convolution_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
ø
b
F__inference_reshape_41_layer_call_and_return_conditional_losses_234895

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
A
input_275
serving_default_input_27:0ÿÿÿÿÿÿÿÿÿF<
dense_130
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿFtensorflow/serving/predict:«
ÈC
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer_with_weights-2

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"ª@
_tf_keras_network@{"class_name": "Functional", "name": "model_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}, "name": "input_27", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_13", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_13", "inbound_nodes": [["input_27", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_39", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_39", "inbound_nodes": [[["tf.expand_dims_13", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_13", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_13", "inbound_nodes": [[["reshape_39", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_40", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_40", "inbound_nodes": [[["fixed_adjacency_graph_convolution_13", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_13", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_13", "inbound_nodes": [[["reshape_40", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_41", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_41", "inbound_nodes": [[["permute_13", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_13", "inbound_nodes": [[["reshape_41", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_27", 0, 0]], "output_layers": [["dense_13", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}, "name": "input_27", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_13", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_13", "inbound_nodes": [["input_27", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_39", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_39", "inbound_nodes": [[["tf.expand_dims_13", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_13", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_13", "inbound_nodes": [[["reshape_39", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_40", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_40", "inbound_nodes": [[["fixed_adjacency_graph_convolution_13", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_13", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_13", "inbound_nodes": [[["reshape_40", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_41", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_41", "inbound_nodes": [[["permute_13", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_13", "inbound_nodes": [[["reshape_41", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_27", 0, 0]], "output_layers": [["dense_13", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input_27", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}}
æ
	keras_api"Ô
_tf_keras_layerº{"class_name": "TFOpLambda", "name": "tf.expand_dims_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_13", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ú
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_39", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
Æ
A

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerþ{"class_name": "FixedAdjacencyGraphConvolution", "name": "fixed_adjacency_graph_convolution_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_adjacency_graph_convolution_13", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}}
ý
regularization_losses
trainable_variables
	variables
 	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "Reshape", "name": "reshape_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_40", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}}

!regularization_losses
"trainable_variables
#	variables
$	keras_api
+&call_and_return_all_conditional_losses
__call__"ò
_tf_keras_layerØ{"class_name": "Permute", "name": "permute_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "permute_13", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+&call_and_return_all_conditional_losses
__call__"é
_tf_keras_layerÏ{"class_name": "Reshape", "name": "reshape_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_41", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}}
Â
)cell
*
state_spec
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+&call_and_return_all_conditional_losses
__call__"

_tf_keras_rnn_layerù	{"class_name": "LSTM", "name": "lstm_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_13", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 70]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 70]}}
é
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
ù

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
ß
9iter

:beta_1

;beta_2
	<decay
=learning_ratemm3m4m>m?m@mvv3v4v>v?v@v"
	optimizer
 "
trackable_list_wrapper
Q
0
1
>2
?3
@4
35
46"
trackable_list_wrapper
X
0
1
2
>3
?4
@5
36
47"
trackable_list_wrapper
Î
regularization_losses

Alayers
trainable_variables
Bnon_trainable_variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
	variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¤serving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses

Flayers
trainable_variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
6:4FF2&fixed_adjacency_graph_convolution_13/A
=:;2+fixed_adjacency_graph_convolution_13/kernel
;:9F2)fixed_adjacency_graph_convolution_13/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
°
regularization_losses

Klayers
trainable_variables
Lnon_trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses

Players
trainable_variables
Qnon_trainable_variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
!regularization_losses

Ulayers
"trainable_variables
Vnon_trainable_variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
#	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
%regularization_losses

Zlayers
&trainable_variables
[non_trainable_variables
\layer_regularization_losses
]metrics
^layer_metrics
'	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
­

>kernel
?recurrent_kernel
@bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"ð
_tf_keras_layerÖ{"class_name": "LSTMCell", "name": "lstm_cell_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_13", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
¼
+regularization_losses

clayers
,trainable_variables
dnon_trainable_variables
elayer_regularization_losses
fmetrics
glayer_metrics
-	variables

hstates
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
/regularization_losses

ilayers
0trainable_variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
1	variables
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
!:dF2dense_13/kernel
:F2dense_13/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
5regularization_losses

nlayers
6trainable_variables
onon_trainable_variables
player_regularization_losses
qmetrics
rlayer_metrics
7	variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	F2lstm_13/lstm_cell_13/kernel
8:6	d2%lstm_13/lstm_cell_13/recurrent_kernel
(:&2lstm_13/lstm_cell_13/bias
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
°
_regularization_losses

ulayers
`trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
ylayer_metrics
a	variables
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	ztotal
	{count
|	variables
}	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ö
	~total
	count

_fn_kwargs
	variables
	keras_api"¬
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
z0
{1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
B:@22Adam/fixed_adjacency_graph_convolution_13/kernel/m
@:>F20Adam/fixed_adjacency_graph_convolution_13/bias/m
&:$dF2Adam/dense_13/kernel/m
 :F2Adam/dense_13/bias/m
3:1	F2"Adam/lstm_13/lstm_cell_13/kernel/m
=:;	d2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+2 Adam/lstm_13/lstm_cell_13/bias/m
B:@22Adam/fixed_adjacency_graph_convolution_13/kernel/v
@:>F20Adam/fixed_adjacency_graph_convolution_13/bias/v
&:$dF2Adam/dense_13/kernel/v
 :F2Adam/dense_13/bias/v
3:1	F2"Adam/lstm_13/lstm_cell_13/kernel/v
=:;	d2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+2 Adam/lstm_13/lstm_cell_13/bias/v
Þ2Û
D__inference_model_13_layer_call_and_return_conditional_losses_235296
D__inference_model_13_layer_call_and_return_conditional_losses_235951
D__inference_model_13_layer_call_and_return_conditional_losses_235709
D__inference_model_13_layer_call_and_return_conditional_losses_235326À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
!__inference__wrapped_model_234136»
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_27ÿÿÿÿÿÿÿÿÿF
ò2ï
)__inference_model_13_layer_call_fn_235972
)__inference_model_13_layer_call_fn_235993
)__inference_model_13_layer_call_fn_235429
)__inference_model_13_layer_call_fn_235378À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_reshape_39_layer_call_and_return_conditional_losses_236006¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_reshape_39_layer_call_fn_236011¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_236064¤
²
FullArgSpec
args
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
E__inference_fixed_adjacency_graph_convolution_13_layer_call_fn_236075¤
²
FullArgSpec
args
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_reshape_40_layer_call_and_return_conditional_losses_236089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_reshape_40_layer_call_fn_236094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
F__inference_permute_13_layer_call_and_return_conditional_losses_234143à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
+__inference_permute_13_layer_call_fn_234149à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_reshape_41_layer_call_and_return_conditional_losses_236107¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_reshape_41_layer_call_fn_236112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
C__inference_lstm_13_layer_call_and_return_conditional_losses_236265
C__inference_lstm_13_layer_call_and_return_conditional_losses_236593
C__inference_lstm_13_layer_call_and_return_conditional_losses_236746
C__inference_lstm_13_layer_call_and_return_conditional_losses_236418Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_lstm_13_layer_call_fn_236768
(__inference_lstm_13_layer_call_fn_236757
(__inference_lstm_13_layer_call_fn_236440
(__inference_lstm_13_layer_call_fn_236429Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_13_layer_call_and_return_conditional_losses_236780
F__inference_dropout_13_layer_call_and_return_conditional_losses_236785´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_13_layer_call_fn_236795
+__inference_dropout_13_layer_call_fn_236790´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_13_layer_call_and_return_conditional_losses_236806¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_13_layer_call_fn_236815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
$__inference_signature_wrapper_235460input_27"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236848
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236881¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
-__inference_lstm_cell_13_layer_call_fn_236915
-__inference_lstm_cell_13_layer_call_fn_236898¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!__inference__wrapped_model_234136v>?@345¢2
+¢(
&#
input_27ÿÿÿÿÿÿÿÿÿF
ª "3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿF¤
D__inference_dense_13_layer_call_and_return_conditional_losses_236806\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 |
)__inference_dense_13_layer_call_fn_236815O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿF¦
F__inference_dropout_13_layer_call_and_return_conditional_losses_236780\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¦
F__inference_dropout_13_layer_call_and_return_conditional_losses_236785\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ~
+__inference_dropout_13_layer_call_fn_236790O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd~
+__inference_dropout_13_layer_call_fn_236795O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿdË
`__inference_fixed_adjacency_graph_convolution_13_layer_call_and_return_conditional_losses_236064g5¢2
+¢(
&#
featuresÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF
 £
E__inference_fixed_adjacency_graph_convolution_13_layer_call_fn_236075Z5¢2
+¢(
&#
featuresÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF´
C__inference_lstm_13_layer_call_and_return_conditional_losses_236265m>?@?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿF

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ´
C__inference_lstm_13_layer_call_and_return_conditional_losses_236418m>?@?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Ä
C__inference_lstm_13_layer_call_and_return_conditional_losses_236593}>?@O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Ä
C__inference_lstm_13_layer_call_and_return_conditional_losses_236746}>?@O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
(__inference_lstm_13_layer_call_fn_236429`>?@?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿF

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_13_layer_call_fn_236440`>?@?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_13_layer_call_fn_236757p>?@O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_lstm_13_layer_call_fn_236768p>?@O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdÊ
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236848ý>?@¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿF
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 Ê
H__inference_lstm_cell_13_layer_call_and_return_conditional_losses_236881ý>?@¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿF
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 
-__inference_lstm_cell_13_layer_call_fn_236898í>?@¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿF
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd
-__inference_lstm_cell_13_layer_call_fn_236915í>?@¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿF
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd¸
D__inference_model_13_layer_call_and_return_conditional_losses_235296p>?@34=¢:
3¢0
&#
input_27ÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ¸
D__inference_model_13_layer_call_and_return_conditional_losses_235326p>?@34=¢:
3¢0
&#
input_27ÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ¶
D__inference_model_13_layer_call_and_return_conditional_losses_235709n>?@34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ¶
D__inference_model_13_layer_call_and_return_conditional_losses_235951n>?@34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 
)__inference_model_13_layer_call_fn_235378c>?@34=¢:
3¢0
&#
input_27ÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
)__inference_model_13_layer_call_fn_235429c>?@34=¢:
3¢0
&#
input_27ÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿF
)__inference_model_13_layer_call_fn_235972a>?@34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p

 
ª "ÿÿÿÿÿÿÿÿÿF
)__inference_model_13_layer_call_fn_235993a>?@34;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿF
p 

 
ª "ÿÿÿÿÿÿÿÿÿFé
F__inference_permute_13_layer_call_and_return_conditional_losses_234143R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_permute_13_layer_call_fn_234149R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ®
F__inference_reshape_39_layer_call_and_return_conditional_losses_236006d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF
 
+__inference_reshape_39_layer_call_fn_236011W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF®
F__inference_reshape_40_layer_call_and_return_conditional_losses_236089d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿF
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿF
 
+__inference_reshape_40_layer_call_fn_236094W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿF
ª " ÿÿÿÿÿÿÿÿÿF®
F__inference_reshape_41_layer_call_and_return_conditional_losses_236107d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª ")¢&

0ÿÿÿÿÿÿÿÿÿF
 
+__inference_reshape_41_layer_call_fn_236112W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF«
$__inference_signature_wrapper_235460>?@34A¢>
¢ 
7ª4
2
input_27&#
input_27ÿÿÿÿÿÿÿÿÿF"3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿF