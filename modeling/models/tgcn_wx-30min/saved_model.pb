̷'
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
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
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??#
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
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
?
*fixed_adjacency_graph_convolution_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*fixed_adjacency_graph_convolution_5/kernel
?
>fixed_adjacency_graph_convolution_5/kernel/Read/ReadVariableOpReadVariableOp*fixed_adjacency_graph_convolution_5/kernel*
_output_shapes

:*
dtype0
?
(fixed_adjacency_graph_convolution_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*9
shared_name*(fixed_adjacency_graph_convolution_5/bias
?
<fixed_adjacency_graph_convolution_5/bias/Read/ReadVariableOpReadVariableOp(fixed_adjacency_graph_convolution_5/bias*
_output_shapes

:F*
dtype0
?
lstm_5/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?**
shared_namelstm_5/lstm_cell_5/kernel
?
-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/kernel*
_output_shapes
:	F?*
dtype0
?
#lstm_5/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*4
shared_name%#lstm_5/lstm_cell_5/recurrent_kernel
?
7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_5/recurrent_kernel*
_output_shapes
:	d?*
dtype0
?
lstm_5/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_5/lstm_cell_5/bias
?
+lstm_5/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/bias*
_output_shapes	
:?*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:dF*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:F*
dtype0
?
%fixed_adjacency_graph_convolution_5/AVarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*6
shared_name'%fixed_adjacency_graph_convolution_5/A
?
9fixed_adjacency_graph_convolution_5/A/Read/ReadVariableOpReadVariableOp%fixed_adjacency_graph_convolution_5/A*
_output_shapes

:FF*
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
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
1Adam/fixed_adjacency_graph_convolution_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/fixed_adjacency_graph_convolution_5/kernel/m
?
EAdam/fixed_adjacency_graph_convolution_5/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_5/kernel/m*
_output_shapes

:*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_5/bias/m
?
CAdam/fixed_adjacency_graph_convolution_5/bias/m/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_5/bias/m*
_output_shapes

:F*
dtype0
?
 Adam/lstm_5/lstm_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/m
?
4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/m*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
?
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/lstm_5/lstm_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_5/bias/m
?
2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:dF*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:F*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
?
1Adam/fixed_adjacency_graph_convolution_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/fixed_adjacency_graph_convolution_5/kernel/v
?
EAdam/fixed_adjacency_graph_convolution_5/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_5/kernel/v*
_output_shapes

:*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_5/bias/v
?
CAdam/fixed_adjacency_graph_convolution_5/bias/v/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_5/bias/v*
_output_shapes

:F*
dtype0
?
 Adam/lstm_5/lstm_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/v
?
4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/v*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
?
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/lstm_5/lstm_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_5/bias/v
?
2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:dF*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:F*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?J
value?JB?J B?J
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
 layer-6
!layer_with_weights-1
!layer-7
"layer-8
#layer_with_weights-2
#layer-9
$trainable_variables
%regularization_losses
&	variables
'	keras_api
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratem?m?-m?.m?/m?0m?1m?2m?3m?v?v?-v?.v?/v?0v?1v?2v?3v?
?
0
1
-2
.3
/4
05
16
27
38
 
F
0
1
-2
.3
44
/5
06
17
28
39
?
5layer_metrics

6layers
trainable_variables
7metrics
8layer_regularization_losses
9non_trainable_variables
regularization_losses
		variables
 
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
:layer_metrics

;layers
trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
regularization_losses
	variables
 
 
 
?
?layer_metrics

@layers
trainable_variables
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
	variables
 
 
 
?
Dlayer_metrics

Elayers
trainable_variables
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
regularization_losses
	variables
 

I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
o
4A

-kernel
.bias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
R
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
R
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
l
^cell
_
state_spec
`trainable_variables
aregularization_losses
b	variables
c	keras_api
R
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
h

2kernel
3bias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
1
-0
.1
/2
03
14
25
36
 
8
-0
.1
42
/3
04
15
26
37
?
llayer_metrics

mlayers
$trainable_variables
nmetrics
olayer_regularization_losses
pnon_trainable_variables
%regularization_losses
&	variables
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
pn
VARIABLE_VALUE*fixed_adjacency_graph_convolution_5/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(fixed_adjacency_graph_convolution_5/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_5/lstm_cell_5/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_5/lstm_cell_5/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_5/lstm_cell_5/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_12/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_12/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%fixed_adjacency_graph_convolution_5/A&variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

q0
r1
 

40
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
?
slayer_metrics

tlayers
Jtrainable_variables
umetrics
vlayer_regularization_losses
wnon_trainable_variables
Kregularization_losses
L	variables

-0
.1
 

-0
.1
42
?
xlayer_metrics

ylayers
Ntrainable_variables
zmetrics
{layer_regularization_losses
|non_trainable_variables
Oregularization_losses
P	variables
 
 
 
?
}layer_metrics

~layers
Rtrainable_variables
metrics
 ?layer_regularization_losses
?non_trainable_variables
Sregularization_losses
T	variables
 
 
 
?
?layer_metrics
?layers
Vtrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
Wregularization_losses
X	variables
 
 
 
?
?layer_metrics
?layers
Ztrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
[regularization_losses
\	variables
?

/kernel
0recurrent_kernel
1bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 

/0
01
12
 

/0
01
12
?
?layer_metrics
?layers
`trainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?states
aregularization_losses
b	variables
 
 
 
?
?layer_metrics
?layers
dtrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
eregularization_losses
f	variables

20
31
 

20
31
?
?layer_metrics
?layers
htrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
iregularization_losses
j	variables
 
F
0
1
2
3
4
5
 6
!7
"8
#9
 
 

40
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 

40
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
/0
01
12
 

/0
01
12
?
?layer_metrics
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?regularization_losses
?	variables
 

^0
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_5/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_5/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_12/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_12/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_5/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_5/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_12/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_12/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_22Placeholder*/
_output_shapes
:?????????F*
dtype0*$
shape:?????????F
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_22dense_13/kerneldense_13/bias%fixed_adjacency_graph_convolution_5/A*fixed_adjacency_graph_convolution_5/kernel(fixed_adjacency_graph_convolution_5/biaslstm_5/lstm_cell_5/kernel#lstm_5/lstm_cell_5/recurrent_kernellstm_5/lstm_cell_5/biasdense_12/kerneldense_12/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_101375
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp>fixed_adjacency_graph_convolution_5/kernel/Read/ReadVariableOp<fixed_adjacency_graph_convolution_5/bias/Read/ReadVariableOp-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_5/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp9fixed_adjacency_graph_convolution_5/A/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_5/kernel/m/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_5/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_5/kernel/v/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_5/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_103668
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate*fixed_adjacency_graph_convolution_5/kernel(fixed_adjacency_graph_convolution_5/biaslstm_5/lstm_cell_5/kernel#lstm_5/lstm_cell_5/recurrent_kernellstm_5/lstm_cell_5/biasdense_12/kerneldense_12/bias%fixed_adjacency_graph_convolution_5/Atotalcounttotal_1count_1Adam/dense_13/kernel/mAdam/dense_13/bias/m1Adam/fixed_adjacency_graph_convolution_5/kernel/m/Adam/fixed_adjacency_graph_convolution_5/bias/m Adam/lstm_5/lstm_cell_5/kernel/m*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mAdam/lstm_5/lstm_cell_5/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/vAdam/dense_13/bias/v1Adam/fixed_adjacency_graph_convolution_5/kernel/v/Adam/fixed_adjacency_graph_convolution_5/bias/v Adam/lstm_5/lstm_cell_5/kernel/v*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vAdam/lstm_5/lstm_cell_5/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/v*1
Tin*
(2&*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_103789ˋ"
?B
?
while_body_100585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_model_15_layer_call_fn_102591

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1009742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
while_cond_100584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100584___redundant_placeholder04
0while_while_cond_100584___redundant_placeholder14
0while_while_cond_100584___redundant_placeholder24
0while_while_cond_100584___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
lstm_5_while_cond_102227*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_102227___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_102227___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_102227___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_102227___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?,
?
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_102683
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?transpose_1/ReadVariableOp?transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:?????????F2
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
unstack?
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
valueB"????F   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2	
Reshape?
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
transpose_1/perm?
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2
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
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2
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
	unstack_2?
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
valueB"????   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_3?
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
transpose_3/perm?
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:?????????2

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
Reshape_5/shape/2?
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape?
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2
	Reshape_5?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
features
?
?
while_cond_103126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103126___redundant_placeholder04
0while_while_cond_103126___redundant_placeholder14
0while_while_cond_103126___redundant_placeholder24
0while_while_cond_103126___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?(
?
D__inference_model_15_layer_call_and_return_conditional_losses_100974

inputs.
*fixed_adjacency_graph_convolution_5_100950.
*fixed_adjacency_graph_convolution_5_100952.
*fixed_adjacency_graph_convolution_5_100954
lstm_5_100960
lstm_5_100962
lstm_5_100964
dense_12_100968
dense_12_100970
identity?? dense_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinputs(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDims?
reshape_29/PartitionedCallPartitionedCall$tf.expand_dims_7/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_29_layer_call_and_return_conditional_losses_1003932
reshape_29/PartitionedCall?
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_29/PartitionedCall:output:0*fixed_adjacency_graph_convolution_5_100950*fixed_adjacency_graph_convolution_5_100952*fixed_adjacency_graph_convolution_5_100954*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_1004542=
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?
reshape_30/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_30_layer_call_and_return_conditional_losses_1004882
reshape_30/PartitionedCall?
permute_7/PartitionedCallPartitionedCall#reshape_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_permute_7_layer_call_and_return_conditional_losses_997582
permute_7/PartitionedCall?
reshape_31/PartitionedCallPartitionedCall"permute_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_31_layer_call_and_return_conditional_losses_1005102
reshape_31/PartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_31/PartitionedCall:output:0lstm_5_100960lstm_5_100962lstm_5_100964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1006702 
lstm_5/StatefulPartitionedCall?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008652$
"dropout_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_12_100968dense_12_100970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1008942"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103467

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????d2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????d2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????d2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????d2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????d2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_101106

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????F*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
G
+__inference_reshape_28_layer_call_fn_102079

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_28_layer_call_and_return_conditional_losses_1011372
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
)__inference_T-GCN-WX_layer_call_fn_101340
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_1013172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
??
?

D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101667

inputs.
*dense_13_tensordot_readvariableop_resource,
(dense_13_biasadd_readvariableop_resourceP
Lmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resourceP
Lmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resourceL
Hmodel_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resource>
:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource@
<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource?
;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource4
0model_15_dense_12_matmul_readvariableop_resource5
1model_15_dense_12_biasadd_readvariableop_resource
identity??dense_13/BiasAdd/ReadVariableOp?!dense_13/Tensordot/ReadVariableOp?(model_15/dense_12/BiasAdd/ReadVariableOp?'model_15/dense_12/MatMul/ReadVariableOp??model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?model_15/lstm_5/while?
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axes?
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_13/Tensordot/freej
dense_13/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_13/Tensordot/Shape?
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axis?
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2?
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axis?
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2_1~
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const?
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod?
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1?
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1?
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axis?
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat?
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stack?
dense_13/Tensordot/transpose	Transposeinputs"dense_13/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
dense_13/Tensordot/transpose?
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_13/Tensordot/Reshape?
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/Tensordot/MatMul?
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2?
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axis?
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1?
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
dense_13/Tensordot?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
dense_13/BiasAddy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_12/dropout/Const?
dropout_12/dropout/MulMuldense_13/BiasAdd:output:0!dropout_12/dropout/Const:output:0*
T0*/
_output_shapes
:?????????F2
dropout_12/dropout/Mul}
dropout_12/dropout/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????F*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform?
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_12/dropout/GreaterEqual/y?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F2!
dropout_12/dropout/GreaterEqual?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F2
dropout_12/dropout/Cast?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F2
dropout_12/dropout/Mul_1p
reshape_28/ShapeShapedropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
reshape_28/Shape?
reshape_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_28/strided_slice/stack?
 reshape_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_28/strided_slice/stack_1?
 reshape_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_28/strided_slice/stack_2?
reshape_28/strided_sliceStridedSlicereshape_28/Shape:output:0'reshape_28/strided_slice/stack:output:0)reshape_28/strided_slice/stack_1:output:0)reshape_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_28/strided_slicez
reshape_28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_28/Reshape/shape/1z
reshape_28/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_28/Reshape/shape/2?
reshape_28/Reshape/shapePack!reshape_28/strided_slice:output:0#reshape_28/Reshape/shape/1:output:0#reshape_28/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_28/Reshape/shape?
reshape_28/ReshapeReshapedropout_12/dropout/Mul_1:z:0!reshape_28/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_28/Reshape?
(model_15/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(model_15/tf.expand_dims_7/ExpandDims/dim?
$model_15/tf.expand_dims_7/ExpandDims
ExpandDimsreshape_28/Reshape:output:01model_15/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2&
$model_15/tf.expand_dims_7/ExpandDims?
model_15/reshape_29/ShapeShape-model_15/tf.expand_dims_7/ExpandDims:output:0*
T0*
_output_shapes
:2
model_15/reshape_29/Shape?
'model_15/reshape_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_29/strided_slice/stack?
)model_15/reshape_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_29/strided_slice/stack_1?
)model_15/reshape_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_29/strided_slice/stack_2?
!model_15/reshape_29/strided_sliceStridedSlice"model_15/reshape_29/Shape:output:00model_15/reshape_29/strided_slice/stack:output:02model_15/reshape_29/strided_slice/stack_1:output:02model_15/reshape_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_29/strided_slice?
#model_15/reshape_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_29/Reshape/shape/1?
#model_15/reshape_29/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_15/reshape_29/Reshape/shape/2?
!model_15/reshape_29/Reshape/shapePack*model_15/reshape_29/strided_slice:output:0,model_15/reshape_29/Reshape/shape/1:output:0,model_15/reshape_29/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_29/Reshape/shape?
model_15/reshape_29/ReshapeReshape-model_15/tf.expand_dims_7/ExpandDims:output:0*model_15/reshape_29/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_15/reshape_29/Reshape?
;model_15/fixed_adjacency_graph_convolution_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;model_15/fixed_adjacency_graph_convolution_5/transpose/perm?
6model_15/fixed_adjacency_graph_convolution_5/transpose	Transpose$model_15/reshape_29/Reshape:output:0Dmodel_15/fixed_adjacency_graph_convolution_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/transpose?
2model_15/fixed_adjacency_graph_convolution_5/ShapeShape:model_15/fixed_adjacency_graph_convolution_5/transpose:y:0*
T0*
_output_shapes
:24
2model_15/fixed_adjacency_graph_convolution_5/Shape?
4model_15/fixed_adjacency_graph_convolution_5/unstackUnpack;model_15/fixed_adjacency_graph_convolution_5/Shape:output:0*
T0*
_output_shapes
: : : *	
num26
4model_15/fixed_adjacency_graph_convolution_5/unstack?
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02E
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOp?
4model_15/fixed_adjacency_graph_convolution_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   26
4model_15/fixed_adjacency_graph_convolution_5/Shape_1?
6model_15/fixed_adjacency_graph_convolution_5/unstack_1Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_1:output:0*
T0*
_output_shapes
: : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_1?
:model_15/fixed_adjacency_graph_convolution_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2<
:model_15/fixed_adjacency_graph_convolution_5/Reshape/shape?
4model_15/fixed_adjacency_graph_convolution_5/ReshapeReshape:model_15/fixed_adjacency_graph_convolution_5/transpose:y:0Cmodel_15/fixed_adjacency_graph_convolution_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F26
4model_15/fixed_adjacency_graph_convolution_5/Reshape?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02I
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?
=model_15/fixed_adjacency_graph_convolution_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_1/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_1	TransposeOmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp:value:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_1?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_1Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_1:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_1?
3model_15/fixed_adjacency_graph_convolution_5/MatMulMatMul=model_15/fixed_adjacency_graph_convolution_5/Reshape:output:0?model_15/fixed_adjacency_graph_convolution_5/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F25
3model_15/fixed_adjacency_graph_convolution_5/MatMul?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shapePack=model_15/fixed_adjacency_graph_convolution_5/unstack:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_2Reshape=model_15/fixed_adjacency_graph_convolution_5/MatMul:product:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_2/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_2	Transpose?model_15/fixed_adjacency_graph_convolution_5/Reshape_2:output:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_2?
4model_15/fixed_adjacency_graph_convolution_5/Shape_2Shape<model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0*
T0*
_output_shapes
:26
4model_15/fixed_adjacency_graph_convolution_5/Shape_2?
6model_15/fixed_adjacency_graph_convolution_5/unstack_2Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_2?
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02E
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOp?
4model_15/fixed_adjacency_graph_convolution_5/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      26
4model_15/fixed_adjacency_graph_convolution_5/Shape_3?
6model_15/fixed_adjacency_graph_convolution_5/unstack_3Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_3:output:0*
T0*
_output_shapes
: : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_3?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_3Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_3?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02I
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?
=model_15/fixed_adjacency_graph_convolution_5/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_3/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_3	TransposeOmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp:value:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/perm:output:0*
T0*
_output_shapes

:2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_3?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_4Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_3:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape:output:0*
T0*
_output_shapes

:28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_4?
5model_15/fixed_adjacency_graph_convolution_5/MatMul_1MatMul?model_15/fixed_adjacency_graph_convolution_5/Reshape_3:output:0?model_15/fixed_adjacency_graph_convolution_5/Reshape_4:output:0*
T0*'
_output_shapes
:?????????27
5model_15/fixed_adjacency_graph_convolution_5/MatMul_1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shapePack?model_15/fixed_adjacency_graph_convolution_5/unstack_2:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_5Reshape?model_15/fixed_adjacency_graph_convolution_5/MatMul_1:product:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_5?
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpReadVariableOpHmodel_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resource*
_output_shapes

:F*
dtype02A
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?
0model_15/fixed_adjacency_graph_convolution_5/addAddV2?model_15/fixed_adjacency_graph_convolution_5/Reshape_5:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F22
0model_15/fixed_adjacency_graph_convolution_5/add?
model_15/reshape_30/ShapeShape4model_15/fixed_adjacency_graph_convolution_5/add:z:0*
T0*
_output_shapes
:2
model_15/reshape_30/Shape?
'model_15/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_30/strided_slice/stack?
)model_15/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_30/strided_slice/stack_1?
)model_15/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_30/strided_slice/stack_2?
!model_15/reshape_30/strided_sliceStridedSlice"model_15/reshape_30/Shape:output:00model_15/reshape_30/strided_slice/stack:output:02model_15/reshape_30/strided_slice/stack_1:output:02model_15/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_30/strided_slice?
#model_15/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_30/Reshape/shape/1?
#model_15/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model_15/reshape_30/Reshape/shape/2?
#model_15/reshape_30/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_15/reshape_30/Reshape/shape/3?
!model_15/reshape_30/Reshape/shapePack*model_15/reshape_30/strided_slice:output:0,model_15/reshape_30/Reshape/shape/1:output:0,model_15/reshape_30/Reshape/shape/2:output:0,model_15/reshape_30/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_30/Reshape/shape?
model_15/reshape_30/ReshapeReshape4model_15/fixed_adjacency_graph_convolution_5/add:z:0*model_15/reshape_30/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
model_15/reshape_30/Reshape?
!model_15/permute_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!model_15/permute_7/transpose/perm?
model_15/permute_7/transpose	Transpose$model_15/reshape_30/Reshape:output:0*model_15/permute_7/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
model_15/permute_7/transpose?
model_15/reshape_31/ShapeShape model_15/permute_7/transpose:y:0*
T0*
_output_shapes
:2
model_15/reshape_31/Shape?
'model_15/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_31/strided_slice/stack?
)model_15/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_31/strided_slice/stack_1?
)model_15/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_31/strided_slice/stack_2?
!model_15/reshape_31/strided_sliceStridedSlice"model_15/reshape_31/Shape:output:00model_15/reshape_31/strided_slice/stack:output:02model_15/reshape_31/strided_slice/stack_1:output:02model_15/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_31/strided_slice?
#model_15/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model_15/reshape_31/Reshape/shape/1?
#model_15/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_31/Reshape/shape/2?
!model_15/reshape_31/Reshape/shapePack*model_15/reshape_31/strided_slice:output:0,model_15/reshape_31/Reshape/shape/1:output:0,model_15/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_31/Reshape/shape?
model_15/reshape_31/ReshapeReshape model_15/permute_7/transpose:y:0*model_15/reshape_31/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_15/reshape_31/Reshape?
model_15/lstm_5/ShapeShape$model_15/reshape_31/Reshape:output:0*
T0*
_output_shapes
:2
model_15/lstm_5/Shape?
#model_15/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_15/lstm_5/strided_slice/stack?
%model_15/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_15/lstm_5/strided_slice/stack_1?
%model_15/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_15/lstm_5/strided_slice/stack_2?
model_15/lstm_5/strided_sliceStridedSlicemodel_15/lstm_5/Shape:output:0,model_15/lstm_5/strided_slice/stack:output:0.model_15/lstm_5/strided_slice/stack_1:output:0.model_15/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_15/lstm_5/strided_slice|
model_15/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
model_15/lstm_5/zeros/mul/y?
model_15/lstm_5/zeros/mulMul&model_15/lstm_5/strided_slice:output:0$model_15/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros/mul
model_15/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_15/lstm_5/zeros/Less/y?
model_15/lstm_5/zeros/LessLessmodel_15/lstm_5/zeros/mul:z:0%model_15/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros/Less?
model_15/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2 
model_15/lstm_5/zeros/packed/1?
model_15/lstm_5/zeros/packedPack&model_15/lstm_5/strided_slice:output:0'model_15/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_15/lstm_5/zeros/packed
model_15/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/zeros/Const?
model_15/lstm_5/zerosFill%model_15/lstm_5/zeros/packed:output:0$model_15/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_15/lstm_5/zeros?
model_15/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
model_15/lstm_5/zeros_1/mul/y?
model_15/lstm_5/zeros_1/mulMul&model_15/lstm_5/strided_slice:output:0&model_15/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros_1/mul?
model_15/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
model_15/lstm_5/zeros_1/Less/y?
model_15/lstm_5/zeros_1/LessLessmodel_15/lstm_5/zeros_1/mul:z:0'model_15/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros_1/Less?
 model_15/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 model_15/lstm_5/zeros_1/packed/1?
model_15/lstm_5/zeros_1/packedPack&model_15/lstm_5/strided_slice:output:0)model_15/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_15/lstm_5/zeros_1/packed?
model_15/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/zeros_1/Const?
model_15/lstm_5/zeros_1Fill'model_15/lstm_5/zeros_1/packed:output:0&model_15/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_15/lstm_5/zeros_1?
model_15/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_15/lstm_5/transpose/perm?
model_15/lstm_5/transpose	Transpose$model_15/reshape_31/Reshape:output:0'model_15/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
model_15/lstm_5/transpose
model_15/lstm_5/Shape_1Shapemodel_15/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
model_15/lstm_5/Shape_1?
%model_15/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_15/lstm_5/strided_slice_1/stack?
'model_15/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_1/stack_1?
'model_15/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_1/stack_2?
model_15/lstm_5/strided_slice_1StridedSlice model_15/lstm_5/Shape_1:output:0.model_15/lstm_5/strided_slice_1/stack:output:00model_15/lstm_5/strided_slice_1/stack_1:output:00model_15/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_15/lstm_5/strided_slice_1?
+model_15/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+model_15/lstm_5/TensorArrayV2/element_shape?
model_15/lstm_5/TensorArrayV2TensorListReserve4model_15/lstm_5/TensorArrayV2/element_shape:output:0(model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_15/lstm_5/TensorArrayV2?
Emodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2G
Emodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
7model_15/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_15/lstm_5/transpose:y:0Nmodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor?
%model_15/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_15/lstm_5/strided_slice_2/stack?
'model_15/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_2/stack_1?
'model_15/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_2/stack_2?
model_15/lstm_5/strided_slice_2StridedSlicemodel_15/lstm_5/transpose:y:0.model_15/lstm_5/strided_slice_2/stack:output:00model_15/lstm_5/strided_slice_2/stack_1:output:00model_15/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2!
model_15/lstm_5/strided_slice_2?
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype023
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
"model_15/lstm_5/lstm_cell_5/MatMulMatMul(model_15/lstm_5/strided_slice_2:output:09model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_15/lstm_5/lstm_cell_5/MatMul?
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype025
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
$model_15/lstm_5/lstm_cell_5/MatMul_1MatMulmodel_15/lstm_5/zeros:output:0;model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_15/lstm_5/lstm_cell_5/MatMul_1?
model_15/lstm_5/lstm_cell_5/addAddV2,model_15/lstm_5/lstm_cell_5/MatMul:product:0.model_15/lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2!
model_15/lstm_5/lstm_cell_5/add?
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
#model_15/lstm_5/lstm_cell_5/BiasAddBiasAdd#model_15/lstm_5/lstm_cell_5/add:z:0:model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_15/lstm_5/lstm_cell_5/BiasAdd?
!model_15/lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_15/lstm_5/lstm_cell_5/Const?
+model_15/lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_15/lstm_5/lstm_cell_5/split/split_dim?
!model_15/lstm_5/lstm_cell_5/splitSplit4model_15/lstm_5/lstm_cell_5/split/split_dim:output:0,model_15/lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2#
!model_15/lstm_5/lstm_cell_5/split?
#model_15/lstm_5/lstm_cell_5/SigmoidSigmoid*model_15/lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2%
#model_15/lstm_5/lstm_cell_5/Sigmoid?
%model_15/lstm_5/lstm_cell_5/Sigmoid_1Sigmoid*model_15/lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/lstm_cell_5/Sigmoid_1?
model_15/lstm_5/lstm_cell_5/mulMul)model_15/lstm_5/lstm_cell_5/Sigmoid_1:y:0 model_15/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????d2!
model_15/lstm_5/lstm_cell_5/mul?
 model_15/lstm_5/lstm_cell_5/ReluRelu*model_15/lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/lstm_cell_5/Relu?
!model_15/lstm_5/lstm_cell_5/mul_1Mul'model_15/lstm_5/lstm_cell_5/Sigmoid:y:0.model_15/lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/mul_1?
!model_15/lstm_5/lstm_cell_5/add_1AddV2#model_15/lstm_5/lstm_cell_5/mul:z:0%model_15/lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/add_1?
%model_15/lstm_5/lstm_cell_5/Sigmoid_2Sigmoid*model_15/lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/lstm_cell_5/Sigmoid_2?
"model_15/lstm_5/lstm_cell_5/Relu_1Relu%model_15/lstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"model_15/lstm_5/lstm_cell_5/Relu_1?
!model_15/lstm_5/lstm_cell_5/mul_2Mul)model_15/lstm_5/lstm_cell_5/Sigmoid_2:y:00model_15/lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/mul_2?
-model_15/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2/
-model_15/lstm_5/TensorArrayV2_1/element_shape?
model_15/lstm_5/TensorArrayV2_1TensorListReserve6model_15/lstm_5/TensorArrayV2_1/element_shape:output:0(model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_15/lstm_5/TensorArrayV2_1n
model_15/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_15/lstm_5/time?
(model_15/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(model_15/lstm_5/while/maximum_iterations?
"model_15/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_15/lstm_5/while/loop_counter?
model_15/lstm_5/whileWhile+model_15/lstm_5/while/loop_counter:output:01model_15/lstm_5/while/maximum_iterations:output:0model_15/lstm_5/time:output:0(model_15/lstm_5/TensorArrayV2_1:handle:0model_15/lstm_5/zeros:output:0 model_15/lstm_5/zeros_1:output:0(model_15/lstm_5/strided_slice_1:output:0Gmodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!model_15_lstm_5_while_body_101567*-
cond%R#
!model_15_lstm_5_while_cond_101566*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
model_15/lstm_5/while?
@model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2B
@model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
2model_15/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStackmodel_15/lstm_5/while:output:3Imodel_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype024
2model_15/lstm_5/TensorArrayV2Stack/TensorListStack?
%model_15/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model_15/lstm_5/strided_slice_3/stack?
'model_15/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/lstm_5/strided_slice_3/stack_1?
'model_15/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_3/stack_2?
model_15/lstm_5/strided_slice_3StridedSlice;model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0.model_15/lstm_5/strided_slice_3/stack:output:00model_15/lstm_5/strided_slice_3/stack_1:output:00model_15/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2!
model_15/lstm_5/strided_slice_3?
 model_15/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_15/lstm_5/transpose_1/perm?
model_15/lstm_5/transpose_1	Transpose;model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0)model_15/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
model_15/lstm_5/transpose_1?
model_15/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/runtime?
!model_15/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2#
!model_15/dropout_13/dropout/Const?
model_15/dropout_13/dropout/MulMul(model_15/lstm_5/strided_slice_3:output:0*model_15/dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2!
model_15/dropout_13/dropout/Mul?
!model_15/dropout_13/dropout/ShapeShape(model_15/lstm_5/strided_slice_3:output:0*
T0*
_output_shapes
:2#
!model_15/dropout_13/dropout/Shape?
8model_15/dropout_13/dropout/random_uniform/RandomUniformRandomUniform*model_15/dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02:
8model_15/dropout_13/dropout/random_uniform/RandomUniform?
*model_15/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2,
*model_15/dropout_13/dropout/GreaterEqual/y?
(model_15/dropout_13/dropout/GreaterEqualGreaterEqualAmodel_15/dropout_13/dropout/random_uniform/RandomUniform:output:03model_15/dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2*
(model_15/dropout_13/dropout/GreaterEqual?
 model_15/dropout_13/dropout/CastCast,model_15/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2"
 model_15/dropout_13/dropout/Cast?
!model_15/dropout_13/dropout/Mul_1Mul#model_15/dropout_13/dropout/Mul:z:0$model_15/dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2#
!model_15/dropout_13/dropout/Mul_1?
'model_15/dense_12/MatMul/ReadVariableOpReadVariableOp0model_15_dense_12_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02)
'model_15/dense_12/MatMul/ReadVariableOp?
model_15/dense_12/MatMulMatMul%model_15/dropout_13/dropout/Mul_1:z:0/model_15/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/MatMul?
(model_15/dense_12/BiasAdd/ReadVariableOpReadVariableOp1model_15_dense_12_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02*
(model_15/dense_12/BiasAdd/ReadVariableOp?
model_15/dense_12/BiasAddBiasAdd"model_15/dense_12/MatMul:product:00model_15/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/BiasAdd?
model_15/dense_12/SigmoidSigmoid"model_15/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/Sigmoid?
IdentityIdentitymodel_15/dense_12/Sigmoid:y:0 ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp)^model_15/dense_12/BiasAdd/ReadVariableOp(^model_15/dense_12/MatMul/ReadVariableOp@^model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpH^model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpH^model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp3^model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2^model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp4^model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp^model_15/lstm_5/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2T
(model_15/dense_12/BiasAdd/ReadVariableOp(model_15/dense_12/BiasAdd/ReadVariableOp2R
'model_15/dense_12/MatMul/ReadVariableOp'model_15/dense_12/MatMul/ReadVariableOp2?
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp2?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpGmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp2?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpGmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp2h
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2f
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp2j
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp2.
model_15/lstm_5/whilemodel_15/lstm_5/while:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_101375
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_997512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
?
?
D__inference_dense_13_layer_call_and_return_conditional_losses_101078

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
)__inference_model_15_layer_call_fn_102612

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1010252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
F__inference_reshape_28_layer_call_and_return_conditional_losses_101137

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
strided_slice/stack_2?
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
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_101111

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????F2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????F2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103212

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
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
:?????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_103127*
condR
while_cond_103126*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
!model_15_lstm_5_while_cond_101851<
8model_15_lstm_5_while_model_15_lstm_5_while_loop_counterB
>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations%
!model_15_lstm_5_while_placeholder'
#model_15_lstm_5_while_placeholder_1'
#model_15_lstm_5_while_placeholder_2'
#model_15_lstm_5_while_placeholder_3>
:model_15_lstm_5_while_less_model_15_lstm_5_strided_slice_1T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101851___redundant_placeholder0T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101851___redundant_placeholder1T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101851___redundant_placeholder2T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101851___redundant_placeholder3"
model_15_lstm_5_while_identity
?
model_15/lstm_5/while/LessLess!model_15_lstm_5_while_placeholder:model_15_lstm_5_while_less_model_15_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
model_15/lstm_5/while/Less?
model_15/lstm_5/while/IdentityIdentitymodel_15/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_15/lstm_5/while/Identity"I
model_15_lstm_5_while_identity'model_15/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
b
F__inference_reshape_28_layer_call_and_return_conditional_losses_102074

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
strided_slice/stack_2?
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
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?&
?
D__inference_model_15_layer_call_and_return_conditional_losses_101025

inputs.
*fixed_adjacency_graph_convolution_5_101001.
*fixed_adjacency_graph_convolution_5_101003.
*fixed_adjacency_graph_convolution_5_101005
lstm_5_101011
lstm_5_101013
lstm_5_101015
dense_12_101019
dense_12_101021
identity?? dense_12/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinputs(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDims?
reshape_29/PartitionedCallPartitionedCall$tf.expand_dims_7/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_29_layer_call_and_return_conditional_losses_1003932
reshape_29/PartitionedCall?
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_29/PartitionedCall:output:0*fixed_adjacency_graph_convolution_5_101001*fixed_adjacency_graph_convolution_5_101003*fixed_adjacency_graph_convolution_5_101005*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_1004542=
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?
reshape_30/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_30_layer_call_and_return_conditional_losses_1004882
reshape_30/PartitionedCall?
permute_7/PartitionedCallPartitionedCall#reshape_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_permute_7_layer_call_and_return_conditional_losses_997582
permute_7/PartitionedCall?
reshape_31/PartitionedCallPartitionedCall"permute_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_31_layer_call_and_return_conditional_losses_1005102
reshape_31/PartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_31/PartitionedCall:output:0lstm_5_101011lstm_5_101013lstm_5_101015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1008232 
lstm_5/StatefulPartitionedCall?
dropout_13/PartitionedCallPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008702
dropout_13/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_12_101019dense_12_101021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1008942"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?$
?
while_body_100164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5_100188_0
while_lstm_cell_5_100190_0
while_lstm_cell_5_100192_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_100188
while_lstm_cell_5_100190
while_lstm_cell_5_100192??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_100188_0while_lstm_cell_5_100190_0while_lstm_cell_5_100192_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998372+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_5_100188while_lstm_cell_5_100188_0"6
while_lstm_cell_5_100190while_lstm_cell_5_100190_0"6
while_lstm_cell_5_100192while_lstm_cell_5_100192_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_100295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100295___redundant_placeholder04
0while_while_cond_100295___redundant_placeholder14
0while_while_cond_100295___redundant_placeholder24
0while_while_cond_100295___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_103279
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_103279___redundant_placeholder04
0while_while_cond_103279___redundant_placeholder14
0while_while_cond_103279___redundant_placeholder24
0while_while_cond_103279___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_102798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_102798___redundant_placeholder04
0while_while_cond_102798___redundant_placeholder14
0while_while_cond_102798___redundant_placeholder24
0while_while_cond_102798___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_100865

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?R
?
__inference__traced_save_103668
file_prefix.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopI
Esavev2_fixed_adjacency_graph_convolution_5_kernel_read_readvariableopG
Csavev2_fixed_adjacency_graph_convolution_5_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableopD
@savev2_fixed_adjacency_graph_convolution_5_a_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_5_kernel_m_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_5_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_5_kernel_v_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_5_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopEsavev2_fixed_adjacency_graph_convolution_5_kernel_read_readvariableopCsavev2_fixed_adjacency_graph_convolution_5_bias_read_readvariableop4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop@savev2_fixed_adjacency_graph_convolution_5_a_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_5_kernel_m_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_5_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_5_kernel_v_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_5_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : ::F:	F?:	d?:?:dF:F:FF: : : : ::::F:	F?:	d?:?:dF:F::::F:	F?:	d?:?:dF:F: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$	 

_output_shapes

:F:%
!

_output_shapes
:	F?:%!

_output_shapes
:	d?:!

_output_shapes	
:?:$ 

_output_shapes

:dF: 

_output_shapes
:F:$ 

_output_shapes

:FF:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:F:%!

_output_shapes
:	F?:%!

_output_shapes
:	d?:!

_output_shapes	
:?:$ 

_output_shapes

:dF: 

_output_shapes
:F:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$  

_output_shapes

:F:%!!

_output_shapes
:	F?:%"!

_output_shapes
:	d?:!#

_output_shapes	
:?:$$ 

_output_shapes

:dF: %

_output_shapes
:F:&

_output_shapes
: 
?	
?
lstm_5_while_cond_102476*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_102476___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_102476___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_102476___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_102476___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_model_15_layer_call_fn_101044
input_23
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1010252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_23
?
b
F__inference_reshape_30_layer_call_and_return_conditional_losses_102708

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
strided_slice/stack_2?
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
?????????2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????F2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101205
input_22
dense_13_101089
dense_13_101091
model_15_101187
model_15_101189
model_15_101191
model_15_101193
model_15_101195
model_15_101197
model_15_101199
model_15_101201
identity?? dense_13/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall? model_15/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_13_101089dense_13_101091*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1010782"
 dense_13/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011062$
"dropout_12/StatefulPartitionedCall?
reshape_28/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_28_layer_call_and_return_conditional_losses_1011372
reshape_28/PartitionedCall?
 model_15/StatefulPartitionedCallStatefulPartitionedCall#reshape_28/PartitionedCall:output:0model_15_101187model_15_101189model_15_101191model_15_101193model_15_101195model_15_101197model_15_101199model_15_101201*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1009742"
 model_15/StatefulPartitionedCall?
IdentityIdentity)model_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall!^model_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2D
 model_15/StatefulPartitionedCall model_15/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
?$
?
while_body_100296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5_100320_0
while_lstm_cell_5_100322_0
while_lstm_cell_5_100324_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_100320
while_lstm_cell_5_100322
while_lstm_cell_5_100324??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_100320_0while_lstm_cell_5_100322_0while_lstm_cell_5_100324_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998702+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_5_100320while_lstm_cell_5_100320_0"6
while_lstm_cell_5_100322while_lstm_cell_5_100322_0"6
while_lstm_cell_5_100324while_lstm_cell_5_100324_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_5_layer_call_fn_103534

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?
?
'__inference_lstm_5_layer_call_fn_103387

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1008232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_102051

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????F2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????F2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_103425

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????F2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_99870

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????d2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????d2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????d2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????d2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????d2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?
?
D__inference_dense_13_layer_call_and_return_conditional_losses_102025

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_100670

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
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
:?????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_100585*
condR
while_cond_100584*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?B
?
while_body_103127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_T-GCN-WX_layer_call_fn_101970

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_1012642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
F__inference_reshape_29_layer_call_and_return_conditional_losses_100393

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
strided_slice/stack_2?
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
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?L
?	
lstm_5_while_body_102477*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_5/MatMul?
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_5/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_5/MatMul_1?
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/MatMul:product:0+lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_5/add?
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd lstm_5/while/lstm_cell_5/add:z:07lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_5/BiasAdd?
lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_5/Const?
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_5/split/split_dim?
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:0)lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2 
lstm_5/while/lstm_cell_5/split?
 lstm_5/while/lstm_cell_5/SigmoidSigmoid'lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2"
 lstm_5/while/lstm_cell_5/Sigmoid?
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2$
"lstm_5/while/lstm_cell_5/Sigmoid_1?
lstm_5/while/lstm_cell_5/mulMul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????d2
lstm_5/while/lstm_cell_5/mul?
lstm_5/while/lstm_cell_5/ReluRelu'lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_5/while/lstm_cell_5/Relu?
lstm_5/while/lstm_cell_5/mul_1Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0+lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/mul_1?
lstm_5/while/lstm_cell_5/add_1AddV2 lstm_5/while/lstm_cell_5/mul:z:0"lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/add_1?
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2$
"lstm_5/while/lstm_cell_5/Sigmoid_2?
lstm_5/while/lstm_cell_5/Relu_1Relu"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2!
lstm_5/while/lstm_cell_5/Relu_1?
lstm_5/while/lstm_cell_5/mul_2Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0-lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1?
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations0^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1?
lstm_5/while/Identity_2Identitylstm_5/while/add:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_2:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_1:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2b
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp2`
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp2d
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
??
?
D__inference_model_15_layer_call_and_return_conditional_losses_102570

inputsG
Cfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_5_add_readvariableop_resource5
1lstm_5_lstm_cell_5_matmul_readvariableop_resource7
3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_5_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_5/add/ReadVariableOp?>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?lstm_5/while?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinputs(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDimsx
reshape_29/ShapeShape$tf.expand_dims_7/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_29/Shape?
reshape_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_29/strided_slice/stack?
 reshape_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_29/strided_slice/stack_1?
 reshape_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_29/strided_slice/stack_2?
reshape_29/strided_sliceStridedSlicereshape_29/Shape:output:0'reshape_29/strided_slice/stack:output:0)reshape_29/strided_slice/stack_1:output:0)reshape_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_29/strided_slicez
reshape_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_29/Reshape/shape/1z
reshape_29/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_29/Reshape/shape/2?
reshape_29/Reshape/shapePack!reshape_29/strided_slice:output:0#reshape_29/Reshape/shape/1:output:0#reshape_29/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_29/Reshape/shape?
reshape_29/ReshapeReshape$tf.expand_dims_7/ExpandDims:output:0!reshape_29/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_29/Reshape?
2fixed_adjacency_graph_convolution_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_5/transpose/perm?
-fixed_adjacency_graph_convolution_5/transpose	Transposereshape_29/Reshape:output:0;fixed_adjacency_graph_convolution_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/transpose?
)fixed_adjacency_graph_convolution_5/ShapeShape1fixed_adjacency_graph_convolution_5/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_5/Shape?
+fixed_adjacency_graph_convolution_5/unstackUnpack2fixed_adjacency_graph_convolution_5/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_5/unstack?
:fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_5/Shape_1?
-fixed_adjacency_graph_convolution_5/unstack_1Unpack4fixed_adjacency_graph_convolution_5/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_1?
1fixed_adjacency_graph_convolution_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_5/Reshape/shape?
+fixed_adjacency_graph_convolution_5/ReshapeReshape1fixed_adjacency_graph_convolution_5/transpose:y:0:fixed_adjacency_graph_convolution_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_5/Reshape?
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_5/transpose_1/perm?
/fixed_adjacency_graph_convolution_5/transpose_1	TransposeFfixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_5/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_5/transpose_1?
3fixed_adjacency_graph_convolution_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_5/Reshape_1/shape?
-fixed_adjacency_graph_convolution_5/Reshape_1Reshape3fixed_adjacency_graph_convolution_5/transpose_1:y:0<fixed_adjacency_graph_convolution_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_5/Reshape_1?
*fixed_adjacency_graph_convolution_5/MatMulMatMul4fixed_adjacency_graph_convolution_5/Reshape:output:06fixed_adjacency_graph_convolution_5/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_5/MatMul?
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_5/Reshape_2/shapePack4fixed_adjacency_graph_convolution_5/unstack:output:0>fixed_adjacency_graph_convolution_5/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_5/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_5/Reshape_2/shape?
-fixed_adjacency_graph_convolution_5/Reshape_2Reshape4fixed_adjacency_graph_convolution_5/MatMul:product:0<fixed_adjacency_graph_convolution_5/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/Reshape_2?
4fixed_adjacency_graph_convolution_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_5/transpose_2/perm?
/fixed_adjacency_graph_convolution_5/transpose_2	Transpose6fixed_adjacency_graph_convolution_5/Reshape_2:output:0=fixed_adjacency_graph_convolution_5/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_5/transpose_2?
+fixed_adjacency_graph_convolution_5/Shape_2Shape3fixed_adjacency_graph_convolution_5/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_5/Shape_2?
-fixed_adjacency_graph_convolution_5/unstack_2Unpack4fixed_adjacency_graph_convolution_5/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_2?
:fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02<
:fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_5/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2-
+fixed_adjacency_graph_convolution_5/Shape_3?
-fixed_adjacency_graph_convolution_5/unstack_3Unpack4fixed_adjacency_graph_convolution_5/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_3?
3fixed_adjacency_graph_convolution_5/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_5/Reshape_3/shape?
-fixed_adjacency_graph_convolution_5/Reshape_3Reshape3fixed_adjacency_graph_convolution_5/transpose_2:y:0<fixed_adjacency_graph_convolution_5/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_5/Reshape_3?
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02@
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_5/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_5/transpose_3/perm?
/fixed_adjacency_graph_convolution_5/transpose_3	TransposeFfixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_5/transpose_3/perm:output:0*
T0*
_output_shapes

:21
/fixed_adjacency_graph_convolution_5/transpose_3?
3fixed_adjacency_graph_convolution_5/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_5/Reshape_4/shape?
-fixed_adjacency_graph_convolution_5/Reshape_4Reshape3fixed_adjacency_graph_convolution_5/transpose_3:y:0<fixed_adjacency_graph_convolution_5/Reshape_4/shape:output:0*
T0*
_output_shapes

:2/
-fixed_adjacency_graph_convolution_5/Reshape_4?
,fixed_adjacency_graph_convolution_5/MatMul_1MatMul6fixed_adjacency_graph_convolution_5/Reshape_3:output:06fixed_adjacency_graph_convolution_5/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2.
,fixed_adjacency_graph_convolution_5/MatMul_1?
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_5/Reshape_5/shapePack6fixed_adjacency_graph_convolution_5/unstack_2:output:0>fixed_adjacency_graph_convolution_5/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_5/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_5/Reshape_5/shape?
-fixed_adjacency_graph_convolution_5/Reshape_5Reshape6fixed_adjacency_graph_convolution_5/MatMul_1:product:0<fixed_adjacency_graph_convolution_5/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/Reshape_5?
6fixed_adjacency_graph_convolution_5/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_5_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_5/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_5/addAddV26fixed_adjacency_graph_convolution_5/Reshape_5:output:0>fixed_adjacency_graph_convolution_5/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2)
'fixed_adjacency_graph_convolution_5/add
reshape_30/ShapeShape+fixed_adjacency_graph_convolution_5/add:z:0*
T0*
_output_shapes
:2
reshape_30/Shape?
reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_30/strided_slice/stack?
 reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_30/strided_slice/stack_1?
 reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_30/strided_slice/stack_2?
reshape_30/strided_sliceStridedSlicereshape_30/Shape:output:0'reshape_30/strided_slice/stack:output:0)reshape_30/strided_slice/stack_1:output:0)reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_30/strided_slicez
reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_30/Reshape/shape/1?
reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_30/Reshape/shape/2z
reshape_30/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_30/Reshape/shape/3?
reshape_30/Reshape/shapePack!reshape_30/strided_slice:output:0#reshape_30/Reshape/shape/1:output:0#reshape_30/Reshape/shape/2:output:0#reshape_30/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_30/Reshape/shape?
reshape_30/ReshapeReshape+fixed_adjacency_graph_convolution_5/add:z:0!reshape_30/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
reshape_30/Reshape?
permute_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_7/transpose/perm?
permute_7/transpose	Transposereshape_30/Reshape:output:0!permute_7/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
permute_7/transposek
reshape_31/ShapeShapepermute_7/transpose:y:0*
T0*
_output_shapes
:2
reshape_31/Shape?
reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_31/strided_slice/stack?
 reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_31/strided_slice/stack_1?
 reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_31/strided_slice/stack_2?
reshape_31/strided_sliceStridedSlicereshape_31/Shape:output:0'reshape_31/strided_slice/stack:output:0)reshape_31/strided_slice/stack_1:output:0)reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_31/strided_slice?
reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_31/Reshape/shape/1z
reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_31/Reshape/shape/2?
reshape_31/Reshape/shapePack!reshape_31/strided_slice:output:0#reshape_31/Reshape/shape/1:output:0#reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_31/Reshape/shape?
reshape_31/ReshapeReshapepermute_7/transpose:y:0!reshape_31/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_31/Reshapeg
lstm_5/ShapeShapereshape_31/Reshape:output:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposereshape_31/Reshape:output:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul?
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul_1?
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/MatMul:product:0%lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/add?
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_5/BiasAddBiasAddlstm_5/lstm_cell_5/add:z:01lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/BiasAddv
lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_5/Const?
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_5/split/split_dim?
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0#lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_5/lstm_cell_5/split?
lstm_5/lstm_cell_5/SigmoidSigmoid!lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid?
lstm_5/lstm_cell_5/Sigmoid_1Sigmoid!lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid_1?
lstm_5/lstm_cell_5/mulMul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul?
lstm_5/lstm_cell_5/ReluRelu!lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Relu?
lstm_5/lstm_cell_5/mul_1Mullstm_5/lstm_cell_5/Sigmoid:y:0%lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul_1?
lstm_5/lstm_cell_5/add_1AddV2lstm_5/lstm_cell_5/mul:z:0lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/add_1?
lstm_5/lstm_cell_5/Sigmoid_2Sigmoid!lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid_2?
lstm_5/lstm_cell_5/Relu_1Relulstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Relu_1?
lstm_5/lstm_cell_5/mul_2Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0'lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_5_matmul_readvariableop_resource3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_102477*$
condR
lstm_5_while_cond_102476*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtime?
dropout_13/IdentityIdentitylstm_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
dropout_13/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_13/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_12/Sigmoid?
IdentityIdentitydense_12/Sigmoid:y:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_5/add/ReadVariableOp?^fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp*^lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp)^lstm_5/lstm_cell_5/MatMul/ReadVariableOp+^lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_5/add/ReadVariableOp6fixed_adjacency_graph_convolution_5/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp2V
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2T
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp(lstm_5/lstm_cell_5/MatMul/ReadVariableOp2X
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
G
+__inference_reshape_30_layer_call_fn_102713

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_30_layer_call_and_return_conditional_losses_1004882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
,__inference_lstm_cell_5_layer_call_fn_103517

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_100870

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?Z
?
!model_15_lstm_5_while_body_101567<
8model_15_lstm_5_while_model_15_lstm_5_while_loop_counterB
>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations%
!model_15_lstm_5_while_placeholder'
#model_15_lstm_5_while_placeholder_1'
#model_15_lstm_5_while_placeholder_2'
#model_15_lstm_5_while_placeholder_3;
7model_15_lstm_5_while_model_15_lstm_5_strided_slice_1_0w
smodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0F
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0H
Dmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0G
Cmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"
model_15_lstm_5_while_identity$
 model_15_lstm_5_while_identity_1$
 model_15_lstm_5_while_identity_2$
 model_15_lstm_5_while_identity_3$
 model_15_lstm_5_while_identity_4$
 model_15_lstm_5_while_identity_59
5model_15_lstm_5_while_model_15_lstm_5_strided_slice_1u
qmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensorD
@model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceF
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceE
Amodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
Gmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2I
Gmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
9model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0!model_15_lstm_5_while_placeholderPmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02;
9model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpBmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype029
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
(model_15/lstm_5/while/lstm_cell_5/MatMulMatMul@model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0?model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_15/lstm_5/while/lstm_cell_5/MatMul?
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpDmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02;
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
*model_15/lstm_5/while/lstm_cell_5/MatMul_1MatMul#model_15_lstm_5_while_placeholder_2Amodel_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*model_15/lstm_5/while/lstm_cell_5/MatMul_1?
%model_15/lstm_5/while/lstm_cell_5/addAddV22model_15/lstm_5/while/lstm_cell_5/MatMul:product:04model_15/lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%model_15/lstm_5/while/lstm_cell_5/add?
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpCmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02:
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
)model_15/lstm_5/while/lstm_cell_5/BiasAddBiasAdd)model_15/lstm_5/while/lstm_cell_5/add:z:0@model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_15/lstm_5/while/lstm_cell_5/BiasAdd?
'model_15/lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_15/lstm_5/while/lstm_cell_5/Const?
1model_15/lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1model_15/lstm_5/while/lstm_cell_5/split/split_dim?
'model_15/lstm_5/while/lstm_cell_5/splitSplit:model_15/lstm_5/while/lstm_cell_5/split/split_dim:output:02model_15/lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2)
'model_15/lstm_5/while/lstm_cell_5/split?
)model_15/lstm_5/while/lstm_cell_5/SigmoidSigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2+
)model_15/lstm_5/while/lstm_cell_5/Sigmoid?
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2-
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_1?
%model_15/lstm_5/while/lstm_cell_5/mulMul/model_15/lstm_5/while/lstm_cell_5/Sigmoid_1:y:0#model_15_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/while/lstm_cell_5/mul?
&model_15/lstm_5/while/lstm_cell_5/ReluRelu0model_15/lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2(
&model_15/lstm_5/while/lstm_cell_5/Relu?
'model_15/lstm_5/while/lstm_cell_5/mul_1Mul-model_15/lstm_5/while/lstm_cell_5/Sigmoid:y:04model_15/lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/mul_1?
'model_15/lstm_5/while/lstm_cell_5/add_1AddV2)model_15/lstm_5/while/lstm_cell_5/mul:z:0+model_15/lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/add_1?
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2-
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_2?
(model_15/lstm_5/while/lstm_cell_5/Relu_1Relu+model_15/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2*
(model_15/lstm_5/while/lstm_cell_5/Relu_1?
'model_15/lstm_5/while/lstm_cell_5/mul_2Mul/model_15/lstm_5/while/lstm_cell_5/Sigmoid_2:y:06model_15/lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/mul_2?
:model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_15_lstm_5_while_placeholder_1!model_15_lstm_5_while_placeholder+model_15/lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem|
model_15/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_15/lstm_5/while/add/y?
model_15/lstm_5/while/addAddV2!model_15_lstm_5_while_placeholder$model_15/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/while/add?
model_15/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_15/lstm_5/while/add_1/y?
model_15/lstm_5/while/add_1AddV28model_15_lstm_5_while_model_15_lstm_5_while_loop_counter&model_15/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/while/add_1?
model_15/lstm_5/while/IdentityIdentitymodel_15/lstm_5/while/add_1:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_15/lstm_5/while/Identity?
 model_15/lstm_5/while/Identity_1Identity>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations9^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_1?
 model_15/lstm_5/while/Identity_2Identitymodel_15/lstm_5/while/add:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_2?
 model_15/lstm_5/while/Identity_3IdentityJmodel_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_3?
 model_15/lstm_5/while/Identity_4Identity+model_15/lstm_5/while/lstm_cell_5/mul_2:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/while/Identity_4?
 model_15/lstm_5/while/Identity_5Identity+model_15/lstm_5/while/lstm_cell_5/add_1:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/while/Identity_5"I
model_15_lstm_5_while_identity'model_15/lstm_5/while/Identity:output:0"M
 model_15_lstm_5_while_identity_1)model_15/lstm_5/while/Identity_1:output:0"M
 model_15_lstm_5_while_identity_2)model_15/lstm_5/while/Identity_2:output:0"M
 model_15_lstm_5_while_identity_3)model_15/lstm_5/while/Identity_3:output:0"M
 model_15_lstm_5_while_identity_4)model_15/lstm_5/while/Identity_4:output:0"M
 model_15_lstm_5_while_identity_5)model_15/lstm_5/while/Identity_5:output:0"?
Amodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resourceCmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceDmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
@model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceBmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"p
5model_15_lstm_5_while_model_15_lstm_5_strided_slice_17model_15_lstm_5_while_model_15_lstm_5_strided_slice_1_0"?
qmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensorsmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2t
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp2r
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp2v
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_102951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_102951___redundant_placeholder04
0while_while_cond_102951___redundant_placeholder14
0while_while_cond_102951___redundant_placeholder24
0while_while_cond_102951___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?D
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_100365

inputs
lstm_cell_5_100283
lstm_cell_5_100285
lstm_cell_5_100287
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_100283lstm_cell_5_100285lstm_cell_5_100287*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998702%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_100283lstm_cell_5_100285lstm_cell_5_100287*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_100296*
condR
while_cond_100295*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?
?
while_cond_100737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100737___redundant_placeholder04
0while_while_cond_100737___redundant_placeholder14
0while_while_cond_100737___redundant_placeholder24
0while_while_cond_100737___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_102046

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????F*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?B
?
while_body_103280
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_100894

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????F2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
D__inference_model_15_layer_call_and_return_conditional_losses_102328

inputsG
Cfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_5_add_readvariableop_resource5
1lstm_5_lstm_cell_5_matmul_readvariableop_resource7
3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_5_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_5/add/ReadVariableOp?>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?lstm_5/while?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinputs(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDimsx
reshape_29/ShapeShape$tf.expand_dims_7/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_29/Shape?
reshape_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_29/strided_slice/stack?
 reshape_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_29/strided_slice/stack_1?
 reshape_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_29/strided_slice/stack_2?
reshape_29/strided_sliceStridedSlicereshape_29/Shape:output:0'reshape_29/strided_slice/stack:output:0)reshape_29/strided_slice/stack_1:output:0)reshape_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_29/strided_slicez
reshape_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_29/Reshape/shape/1z
reshape_29/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_29/Reshape/shape/2?
reshape_29/Reshape/shapePack!reshape_29/strided_slice:output:0#reshape_29/Reshape/shape/1:output:0#reshape_29/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_29/Reshape/shape?
reshape_29/ReshapeReshape$tf.expand_dims_7/ExpandDims:output:0!reshape_29/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_29/Reshape?
2fixed_adjacency_graph_convolution_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_5/transpose/perm?
-fixed_adjacency_graph_convolution_5/transpose	Transposereshape_29/Reshape:output:0;fixed_adjacency_graph_convolution_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/transpose?
)fixed_adjacency_graph_convolution_5/ShapeShape1fixed_adjacency_graph_convolution_5/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_5/Shape?
+fixed_adjacency_graph_convolution_5/unstackUnpack2fixed_adjacency_graph_convolution_5/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_5/unstack?
:fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_5/Shape_1?
-fixed_adjacency_graph_convolution_5/unstack_1Unpack4fixed_adjacency_graph_convolution_5/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_1?
1fixed_adjacency_graph_convolution_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_5/Reshape/shape?
+fixed_adjacency_graph_convolution_5/ReshapeReshape1fixed_adjacency_graph_convolution_5/transpose:y:0:fixed_adjacency_graph_convolution_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_5/Reshape?
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_5/transpose_1/perm?
/fixed_adjacency_graph_convolution_5/transpose_1	TransposeFfixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_5/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_5/transpose_1?
3fixed_adjacency_graph_convolution_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_5/Reshape_1/shape?
-fixed_adjacency_graph_convolution_5/Reshape_1Reshape3fixed_adjacency_graph_convolution_5/transpose_1:y:0<fixed_adjacency_graph_convolution_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_5/Reshape_1?
*fixed_adjacency_graph_convolution_5/MatMulMatMul4fixed_adjacency_graph_convolution_5/Reshape:output:06fixed_adjacency_graph_convolution_5/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_5/MatMul?
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_5/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_5/Reshape_2/shapePack4fixed_adjacency_graph_convolution_5/unstack:output:0>fixed_adjacency_graph_convolution_5/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_5/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_5/Reshape_2/shape?
-fixed_adjacency_graph_convolution_5/Reshape_2Reshape4fixed_adjacency_graph_convolution_5/MatMul:product:0<fixed_adjacency_graph_convolution_5/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/Reshape_2?
4fixed_adjacency_graph_convolution_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_5/transpose_2/perm?
/fixed_adjacency_graph_convolution_5/transpose_2	Transpose6fixed_adjacency_graph_convolution_5/Reshape_2:output:0=fixed_adjacency_graph_convolution_5/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_5/transpose_2?
+fixed_adjacency_graph_convolution_5/Shape_2Shape3fixed_adjacency_graph_convolution_5/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_5/Shape_2?
-fixed_adjacency_graph_convolution_5/unstack_2Unpack4fixed_adjacency_graph_convolution_5/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_2?
:fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02<
:fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_5/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2-
+fixed_adjacency_graph_convolution_5/Shape_3?
-fixed_adjacency_graph_convolution_5/unstack_3Unpack4fixed_adjacency_graph_convolution_5/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_5/unstack_3?
3fixed_adjacency_graph_convolution_5/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_5/Reshape_3/shape?
-fixed_adjacency_graph_convolution_5/Reshape_3Reshape3fixed_adjacency_graph_convolution_5/transpose_2:y:0<fixed_adjacency_graph_convolution_5/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_5/Reshape_3?
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02@
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_5/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_5/transpose_3/perm?
/fixed_adjacency_graph_convolution_5/transpose_3	TransposeFfixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_5/transpose_3/perm:output:0*
T0*
_output_shapes

:21
/fixed_adjacency_graph_convolution_5/transpose_3?
3fixed_adjacency_graph_convolution_5/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_5/Reshape_4/shape?
-fixed_adjacency_graph_convolution_5/Reshape_4Reshape3fixed_adjacency_graph_convolution_5/transpose_3:y:0<fixed_adjacency_graph_convolution_5/Reshape_4/shape:output:0*
T0*
_output_shapes

:2/
-fixed_adjacency_graph_convolution_5/Reshape_4?
,fixed_adjacency_graph_convolution_5/MatMul_1MatMul6fixed_adjacency_graph_convolution_5/Reshape_3:output:06fixed_adjacency_graph_convolution_5/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2.
,fixed_adjacency_graph_convolution_5/MatMul_1?
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_5/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_5/Reshape_5/shapePack6fixed_adjacency_graph_convolution_5/unstack_2:output:0>fixed_adjacency_graph_convolution_5/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_5/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_5/Reshape_5/shape?
-fixed_adjacency_graph_convolution_5/Reshape_5Reshape6fixed_adjacency_graph_convolution_5/MatMul_1:product:0<fixed_adjacency_graph_convolution_5/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_5/Reshape_5?
6fixed_adjacency_graph_convolution_5/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_5_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_5/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_5/addAddV26fixed_adjacency_graph_convolution_5/Reshape_5:output:0>fixed_adjacency_graph_convolution_5/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2)
'fixed_adjacency_graph_convolution_5/add
reshape_30/ShapeShape+fixed_adjacency_graph_convolution_5/add:z:0*
T0*
_output_shapes
:2
reshape_30/Shape?
reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_30/strided_slice/stack?
 reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_30/strided_slice/stack_1?
 reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_30/strided_slice/stack_2?
reshape_30/strided_sliceStridedSlicereshape_30/Shape:output:0'reshape_30/strided_slice/stack:output:0)reshape_30/strided_slice/stack_1:output:0)reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_30/strided_slicez
reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_30/Reshape/shape/1?
reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_30/Reshape/shape/2z
reshape_30/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_30/Reshape/shape/3?
reshape_30/Reshape/shapePack!reshape_30/strided_slice:output:0#reshape_30/Reshape/shape/1:output:0#reshape_30/Reshape/shape/2:output:0#reshape_30/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_30/Reshape/shape?
reshape_30/ReshapeReshape+fixed_adjacency_graph_convolution_5/add:z:0!reshape_30/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
reshape_30/Reshape?
permute_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_7/transpose/perm?
permute_7/transpose	Transposereshape_30/Reshape:output:0!permute_7/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
permute_7/transposek
reshape_31/ShapeShapepermute_7/transpose:y:0*
T0*
_output_shapes
:2
reshape_31/Shape?
reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_31/strided_slice/stack?
 reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_31/strided_slice/stack_1?
 reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_31/strided_slice/stack_2?
reshape_31/strided_sliceStridedSlicereshape_31/Shape:output:0'reshape_31/strided_slice/stack:output:0)reshape_31/strided_slice/stack_1:output:0)reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_31/strided_slice?
reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_31/Reshape/shape/1z
reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_31/Reshape/shape/2?
reshape_31/Reshape/shapePack!reshape_31/strided_slice:output:0#reshape_31/Reshape/shape/1:output:0#reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_31/Reshape/shape?
reshape_31/ReshapeReshapepermute_7/transpose:y:0!reshape_31/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_31/Reshapeg
lstm_5/ShapeShapereshape_31/Reshape:output:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposereshape_31/Reshape:output:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul?
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02,
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul_1?
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/MatMul:product:0%lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/add?
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_5/BiasAddBiasAddlstm_5/lstm_cell_5/add:z:01lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/BiasAddv
lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_5/Const?
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_5/split/split_dim?
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0#lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_5/lstm_cell_5/split?
lstm_5/lstm_cell_5/SigmoidSigmoid!lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid?
lstm_5/lstm_cell_5/Sigmoid_1Sigmoid!lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid_1?
lstm_5/lstm_cell_5/mulMul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul?
lstm_5/lstm_cell_5/ReluRelu!lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Relu?
lstm_5/lstm_cell_5/mul_1Mullstm_5/lstm_cell_5/Sigmoid:y:0%lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul_1?
lstm_5/lstm_cell_5/add_1AddV2lstm_5/lstm_cell_5/mul:z:0lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/add_1?
lstm_5/lstm_cell_5/Sigmoid_2Sigmoid!lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Sigmoid_2?
lstm_5/lstm_cell_5/Relu_1Relulstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/Relu_1?
lstm_5/lstm_cell_5/mul_2Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0'lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_5/lstm_cell_5/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_5_matmul_readvariableop_resource3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_102228*$
condR
lstm_5_while_cond_102227*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtimey
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_13/dropout/Const?
dropout_13/dropout/MulMullstm_5/strided_slice_3:output:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout_13/dropout/Mul?
dropout_13/dropout/ShapeShapelstm_5/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout_13/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_12/BiasAdd|
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_12/Sigmoid?
IdentityIdentitydense_12/Sigmoid:y:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_5/add/ReadVariableOp?^fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp*^lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp)^lstm_5/lstm_cell_5/MatMul/ReadVariableOp+^lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_5/add/ReadVariableOp6fixed_adjacency_graph_convolution_5/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp2V
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2T
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp(lstm_5/lstm_cell_5/MatMul/ReadVariableOp2X
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
F__inference_reshape_30_layer_call_and_return_conditional_losses_100488

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
strided_slice/stack_2?
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
?????????2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????F2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
F__inference_reshape_31_layer_call_and_return_conditional_losses_102726

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
strided_slice/stack_2?
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
?????????2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
while_cond_100163
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100163___redundant_placeholder04
0while_while_cond_100163___redundant_placeholder14
0while_while_cond_100163___redundant_placeholder24
0while_while_cond_100163___redundant_placeholder3
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
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
)__inference_T-GCN-WX_layer_call_fn_101995

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_1013172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
'__inference_lstm_5_layer_call_fn_103048
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1002332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?
?
!model_15_lstm_5_while_cond_101566<
8model_15_lstm_5_while_model_15_lstm_5_while_loop_counterB
>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations%
!model_15_lstm_5_while_placeholder'
#model_15_lstm_5_while_placeholder_1'
#model_15_lstm_5_while_placeholder_2'
#model_15_lstm_5_while_placeholder_3>
:model_15_lstm_5_while_less_model_15_lstm_5_strided_slice_1T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101566___redundant_placeholder0T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101566___redundant_placeholder1T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101566___redundant_placeholder2T
Pmodel_15_lstm_5_while_model_15_lstm_5_while_cond_101566___redundant_placeholder3"
model_15_lstm_5_while_identity
?
model_15/lstm_5/while/LessLess!model_15_lstm_5_while_placeholder:model_15_lstm_5_while_less_model_15_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
model_15/lstm_5/while/Less?
model_15/lstm_5/while/IdentityIdentitymodel_15/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_15/lstm_5/while/Identity"I
model_15_lstm_5_while_identity'model_15/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
b
F__inference_reshape_29_layer_call_and_return_conditional_losses_102625

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
strided_slice/stack_2?
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
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103500

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????d2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????d2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????d2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????d2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????d2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?
`
D__inference_permute_7_layer_call_and_return_conditional_losses_99758

inputs
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
	transpose?
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_dense_13_layer_call_fn_102034

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1010782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?B
?
while_body_102799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?B
?
while_body_100738
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?L
?	
lstm_5_while_body_102228*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_5/MatMul?
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype022
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_5/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_5/MatMul_1?
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/MatMul:product:0+lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_5/add?
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd lstm_5/while/lstm_cell_5/add:z:07lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_5/BiasAdd?
lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_5/Const?
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_5/split/split_dim?
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:0)lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2 
lstm_5/while/lstm_cell_5/split?
 lstm_5/while/lstm_cell_5/SigmoidSigmoid'lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2"
 lstm_5/while/lstm_cell_5/Sigmoid?
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2$
"lstm_5/while/lstm_cell_5/Sigmoid_1?
lstm_5/while/lstm_cell_5/mulMul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????d2
lstm_5/while/lstm_cell_5/mul?
lstm_5/while/lstm_cell_5/ReluRelu'lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_5/while/lstm_cell_5/Relu?
lstm_5/while/lstm_cell_5/mul_1Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0+lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/mul_1?
lstm_5/while/lstm_cell_5/add_1AddV2 lstm_5/while/lstm_cell_5/mul:z:0"lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/add_1?
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2$
"lstm_5/while/lstm_cell_5/Sigmoid_2?
lstm_5/while/lstm_cell_5/Relu_1Relu"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2!
lstm_5/while/lstm_cell_5/Relu_1?
lstm_5/while/lstm_cell_5/mul_2Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0-lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2 
lstm_5/while/lstm_cell_5/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1?
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations0^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1?
lstm_5/while/Identity_2Identitylstm_5/while/add:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_2:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_1:z:00^lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2b
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp2`
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp2d
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_reshape_29_layer_call_fn_102630

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_29_layer_call_and_return_conditional_losses_1003932
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
)__inference_model_15_layer_call_fn_100993
input_23
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1009742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_23
?B
?
while_body_102952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/add_1?
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?Z
?
!model_15_lstm_5_while_body_101852<
8model_15_lstm_5_while_model_15_lstm_5_while_loop_counterB
>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations%
!model_15_lstm_5_while_placeholder'
#model_15_lstm_5_while_placeholder_1'
#model_15_lstm_5_while_placeholder_2'
#model_15_lstm_5_while_placeholder_3;
7model_15_lstm_5_while_model_15_lstm_5_strided_slice_1_0w
smodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0F
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0H
Dmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0G
Cmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"
model_15_lstm_5_while_identity$
 model_15_lstm_5_while_identity_1$
 model_15_lstm_5_while_identity_2$
 model_15_lstm_5_while_identity_3$
 model_15_lstm_5_while_identity_4$
 model_15_lstm_5_while_identity_59
5model_15_lstm_5_while_model_15_lstm_5_strided_slice_1u
qmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensorD
@model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceF
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceE
Amodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
Gmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2I
Gmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
9model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0!model_15_lstm_5_while_placeholderPmodel_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02;
9model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpBmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype029
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
(model_15/lstm_5/while/lstm_cell_5/MatMulMatMul@model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0?model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_15/lstm_5/while/lstm_cell_5/MatMul?
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpDmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02;
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
*model_15/lstm_5/while/lstm_cell_5/MatMul_1MatMul#model_15_lstm_5_while_placeholder_2Amodel_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*model_15/lstm_5/while/lstm_cell_5/MatMul_1?
%model_15/lstm_5/while/lstm_cell_5/addAddV22model_15/lstm_5/while/lstm_cell_5/MatMul:product:04model_15/lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%model_15/lstm_5/while/lstm_cell_5/add?
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpCmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02:
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
)model_15/lstm_5/while/lstm_cell_5/BiasAddBiasAdd)model_15/lstm_5/while/lstm_cell_5/add:z:0@model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_15/lstm_5/while/lstm_cell_5/BiasAdd?
'model_15/lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_15/lstm_5/while/lstm_cell_5/Const?
1model_15/lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1model_15/lstm_5/while/lstm_cell_5/split/split_dim?
'model_15/lstm_5/while/lstm_cell_5/splitSplit:model_15/lstm_5/while/lstm_cell_5/split/split_dim:output:02model_15/lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2)
'model_15/lstm_5/while/lstm_cell_5/split?
)model_15/lstm_5/while/lstm_cell_5/SigmoidSigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2+
)model_15/lstm_5/while/lstm_cell_5/Sigmoid?
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2-
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_1?
%model_15/lstm_5/while/lstm_cell_5/mulMul/model_15/lstm_5/while/lstm_cell_5/Sigmoid_1:y:0#model_15_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/while/lstm_cell_5/mul?
&model_15/lstm_5/while/lstm_cell_5/ReluRelu0model_15/lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2(
&model_15/lstm_5/while/lstm_cell_5/Relu?
'model_15/lstm_5/while/lstm_cell_5/mul_1Mul-model_15/lstm_5/while/lstm_cell_5/Sigmoid:y:04model_15/lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/mul_1?
'model_15/lstm_5/while/lstm_cell_5/add_1AddV2)model_15/lstm_5/while/lstm_cell_5/mul:z:0+model_15/lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/add_1?
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid0model_15/lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2-
+model_15/lstm_5/while/lstm_cell_5/Sigmoid_2?
(model_15/lstm_5/while/lstm_cell_5/Relu_1Relu+model_15/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2*
(model_15/lstm_5/while/lstm_cell_5/Relu_1?
'model_15/lstm_5/while/lstm_cell_5/mul_2Mul/model_15/lstm_5/while/lstm_cell_5/Sigmoid_2:y:06model_15/lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2)
'model_15/lstm_5/while/lstm_cell_5/mul_2?
:model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_15_lstm_5_while_placeholder_1!model_15_lstm_5_while_placeholder+model_15/lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem|
model_15/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_15/lstm_5/while/add/y?
model_15/lstm_5/while/addAddV2!model_15_lstm_5_while_placeholder$model_15/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/while/add?
model_15/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_15/lstm_5/while/add_1/y?
model_15/lstm_5/while/add_1AddV28model_15_lstm_5_while_model_15_lstm_5_while_loop_counter&model_15/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/while/add_1?
model_15/lstm_5/while/IdentityIdentitymodel_15/lstm_5/while/add_1:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_15/lstm_5/while/Identity?
 model_15/lstm_5/while/Identity_1Identity>model_15_lstm_5_while_model_15_lstm_5_while_maximum_iterations9^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_1?
 model_15/lstm_5/while/Identity_2Identitymodel_15/lstm_5/while/add:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_2?
 model_15/lstm_5/while/Identity_3IdentityJmodel_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_15/lstm_5/while/Identity_3?
 model_15/lstm_5/while/Identity_4Identity+model_15/lstm_5/while/lstm_cell_5/mul_2:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/while/Identity_4?
 model_15/lstm_5/while/Identity_5Identity+model_15/lstm_5/while/lstm_cell_5/add_1:z:09^model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8^model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:^model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/while/Identity_5"I
model_15_lstm_5_while_identity'model_15/lstm_5/while/Identity:output:0"M
 model_15_lstm_5_while_identity_1)model_15/lstm_5/while/Identity_1:output:0"M
 model_15_lstm_5_while_identity_2)model_15/lstm_5/while/Identity_2:output:0"M
 model_15_lstm_5_while_identity_3)model_15/lstm_5/while/Identity_3:output:0"M
 model_15_lstm_5_while_identity_4)model_15/lstm_5/while/Identity_4:output:0"M
 model_15_lstm_5_while_identity_5)model_15/lstm_5/while/Identity_5:output:0"?
Amodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resourceCmodel_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
Bmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceDmodel_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
@model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceBmodel_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"p
5model_15_lstm_5_while_model_15_lstm_5_strided_slice_17model_15_lstm_5_while_model_15_lstm_5_strided_slice_1_0"?
qmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensorsmodel_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2t
8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp8model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp2r
7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp7model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp2v
9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp9model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_103399

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?,
?
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_100454
features#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?transpose_1/ReadVariableOp?transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposefeaturestranspose/perm:output:0*
T0*+
_output_shapes
:?????????F2
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
unstack?
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
valueB"????F   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2	
Reshape?
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
transpose_1/perm?
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:FF2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2
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
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2
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
	unstack_2?
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
valueB"????   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_3?
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
transpose_3/perm?
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:?????????2

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
Reshape_5/shape/2?
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape?
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2
	Reshape_5?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2(
add/ReadVariableOpadd/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
features
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_102884
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_102799*
condR
while_cond_102798*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?
?
)T-GCN-WX_model_15_lstm_5_while_cond_99657N
Jt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_loop_counterT
Pt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_maximum_iterations.
*t_gcn_wx_model_15_lstm_5_while_placeholder0
,t_gcn_wx_model_15_lstm_5_while_placeholder_10
,t_gcn_wx_model_15_lstm_5_while_placeholder_20
,t_gcn_wx_model_15_lstm_5_while_placeholder_3P
Lt_gcn_wx_model_15_lstm_5_while_less_t_gcn_wx_model_15_lstm_5_strided_slice_1e
at_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_cond_99657___redundant_placeholder0e
at_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_cond_99657___redundant_placeholder1e
at_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_cond_99657___redundant_placeholder2e
at_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_cond_99657___redundant_placeholder3+
't_gcn_wx_model_15_lstm_5_while_identity
?
#T-GCN-WX/model_15/lstm_5/while/LessLess*t_gcn_wx_model_15_lstm_5_while_placeholderLt_gcn_wx_model_15_lstm_5_while_less_t_gcn_wx_model_15_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_15/lstm_5/while/Less?
'T-GCN-WX/model_15/lstm_5/while/IdentityIdentity'T-GCN-WX/model_15/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2)
'T-GCN-WX/model_15/lstm_5/while/Identity"[
't_gcn_wx_model_15_lstm_5_while_identity0T-GCN-WX/model_15/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
??
?

D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101945

inputs.
*dense_13_tensordot_readvariableop_resource,
(dense_13_biasadd_readvariableop_resourceP
Lmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resourceP
Lmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resourceL
Hmodel_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resource>
:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource@
<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource?
;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource4
0model_15_dense_12_matmul_readvariableop_resource5
1model_15_dense_12_biasadd_readvariableop_resource
identity??dense_13/BiasAdd/ReadVariableOp?!dense_13/Tensordot/ReadVariableOp?(model_15/dense_12/BiasAdd/ReadVariableOp?'model_15/dense_12/MatMul/ReadVariableOp??model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?model_15/lstm_5/while?
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axes?
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_13/Tensordot/freej
dense_13/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_13/Tensordot/Shape?
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axis?
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2?
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axis?
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2_1~
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const?
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod?
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1?
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1?
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axis?
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat?
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stack?
dense_13/Tensordot/transpose	Transposeinputs"dense_13/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
dense_13/Tensordot/transpose?
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_13/Tensordot/Reshape?
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/Tensordot/MatMul?
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2?
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axis?
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1?
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
dense_13/Tensordot?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
dense_13/BiasAdd?
dropout_12/IdentityIdentitydense_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F2
dropout_12/Identityp
reshape_28/ShapeShapedropout_12/Identity:output:0*
T0*
_output_shapes
:2
reshape_28/Shape?
reshape_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_28/strided_slice/stack?
 reshape_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_28/strided_slice/stack_1?
 reshape_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_28/strided_slice/stack_2?
reshape_28/strided_sliceStridedSlicereshape_28/Shape:output:0'reshape_28/strided_slice/stack:output:0)reshape_28/strided_slice/stack_1:output:0)reshape_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_28/strided_slicez
reshape_28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_28/Reshape/shape/1z
reshape_28/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_28/Reshape/shape/2?
reshape_28/Reshape/shapePack!reshape_28/strided_slice:output:0#reshape_28/Reshape/shape/1:output:0#reshape_28/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_28/Reshape/shape?
reshape_28/ReshapeReshapedropout_12/Identity:output:0!reshape_28/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_28/Reshape?
(model_15/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(model_15/tf.expand_dims_7/ExpandDims/dim?
$model_15/tf.expand_dims_7/ExpandDims
ExpandDimsreshape_28/Reshape:output:01model_15/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2&
$model_15/tf.expand_dims_7/ExpandDims?
model_15/reshape_29/ShapeShape-model_15/tf.expand_dims_7/ExpandDims:output:0*
T0*
_output_shapes
:2
model_15/reshape_29/Shape?
'model_15/reshape_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_29/strided_slice/stack?
)model_15/reshape_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_29/strided_slice/stack_1?
)model_15/reshape_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_29/strided_slice/stack_2?
!model_15/reshape_29/strided_sliceStridedSlice"model_15/reshape_29/Shape:output:00model_15/reshape_29/strided_slice/stack:output:02model_15/reshape_29/strided_slice/stack_1:output:02model_15/reshape_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_29/strided_slice?
#model_15/reshape_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_29/Reshape/shape/1?
#model_15/reshape_29/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_15/reshape_29/Reshape/shape/2?
!model_15/reshape_29/Reshape/shapePack*model_15/reshape_29/strided_slice:output:0,model_15/reshape_29/Reshape/shape/1:output:0,model_15/reshape_29/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_29/Reshape/shape?
model_15/reshape_29/ReshapeReshape-model_15/tf.expand_dims_7/ExpandDims:output:0*model_15/reshape_29/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_15/reshape_29/Reshape?
;model_15/fixed_adjacency_graph_convolution_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2=
;model_15/fixed_adjacency_graph_convolution_5/transpose/perm?
6model_15/fixed_adjacency_graph_convolution_5/transpose	Transpose$model_15/reshape_29/Reshape:output:0Dmodel_15/fixed_adjacency_graph_convolution_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/transpose?
2model_15/fixed_adjacency_graph_convolution_5/ShapeShape:model_15/fixed_adjacency_graph_convolution_5/transpose:y:0*
T0*
_output_shapes
:24
2model_15/fixed_adjacency_graph_convolution_5/Shape?
4model_15/fixed_adjacency_graph_convolution_5/unstackUnpack;model_15/fixed_adjacency_graph_convolution_5/Shape:output:0*
T0*
_output_shapes
: : : *	
num26
4model_15/fixed_adjacency_graph_convolution_5/unstack?
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02E
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOp?
4model_15/fixed_adjacency_graph_convolution_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   26
4model_15/fixed_adjacency_graph_convolution_5/Shape_1?
6model_15/fixed_adjacency_graph_convolution_5/unstack_1Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_1:output:0*
T0*
_output_shapes
: : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_1?
:model_15/fixed_adjacency_graph_convolution_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2<
:model_15/fixed_adjacency_graph_convolution_5/Reshape/shape?
4model_15/fixed_adjacency_graph_convolution_5/ReshapeReshape:model_15/fixed_adjacency_graph_convolution_5/transpose:y:0Cmodel_15/fixed_adjacency_graph_convolution_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F26
4model_15/fixed_adjacency_graph_convolution_5/Reshape?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02I
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?
=model_15/fixed_adjacency_graph_convolution_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_1/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_1	TransposeOmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp:value:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_1?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_1Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_1:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_1?
3model_15/fixed_adjacency_graph_convolution_5/MatMulMatMul=model_15/fixed_adjacency_graph_convolution_5/Reshape:output:0?model_15/fixed_adjacency_graph_convolution_5/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F25
3model_15/fixed_adjacency_graph_convolution_5/MatMul?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shapePack=model_15/fixed_adjacency_graph_convolution_5/unstack:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_2Reshape=model_15/fixed_adjacency_graph_convolution_5/MatMul:product:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_2/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_2	Transpose?model_15/fixed_adjacency_graph_convolution_5/Reshape_2:output:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_2?
4model_15/fixed_adjacency_graph_convolution_5/Shape_2Shape<model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0*
T0*
_output_shapes
:26
4model_15/fixed_adjacency_graph_convolution_5/Shape_2?
6model_15/fixed_adjacency_graph_convolution_5/unstack_2Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_2?
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02E
Cmodel_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOp?
4model_15/fixed_adjacency_graph_convolution_5/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      26
4model_15/fixed_adjacency_graph_convolution_5/Shape_3?
6model_15/fixed_adjacency_graph_convolution_5/unstack_3Unpack=model_15/fixed_adjacency_graph_convolution_5/Shape_3:output:0*
T0*
_output_shapes
: : *	
num28
6model_15/fixed_adjacency_graph_convolution_5/unstack_3?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_3Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_3?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpReadVariableOpLmodel_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02I
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?
=model_15/fixed_adjacency_graph_convolution_5/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_15/fixed_adjacency_graph_convolution_5/transpose_3/perm?
8model_15/fixed_adjacency_graph_convolution_5/transpose_3	TransposeOmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp:value:0Fmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/perm:output:0*
T0*
_output_shapes

:2:
8model_15/fixed_adjacency_graph_convolution_5/transpose_3?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_4Reshape<model_15/fixed_adjacency_graph_convolution_5/transpose_3:y:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape:output:0*
T0*
_output_shapes

:28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_4?
5model_15/fixed_adjacency_graph_convolution_5/MatMul_1MatMul?model_15/fixed_adjacency_graph_convolution_5/Reshape_3:output:0?model_15/fixed_adjacency_graph_convolution_5/Reshape_4:output:0*
T0*'
_output_shapes
:?????????27
5model_15/fixed_adjacency_graph_convolution_5/MatMul_1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1?
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2?
<model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shapePack?model_15/fixed_adjacency_graph_convolution_5/unstack_2:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2>
<model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape?
6model_15/fixed_adjacency_graph_convolution_5/Reshape_5Reshape?model_15/fixed_adjacency_graph_convolution_5/MatMul_1:product:0Emodel_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F28
6model_15/fixed_adjacency_graph_convolution_5/Reshape_5?
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpReadVariableOpHmodel_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resource*
_output_shapes

:F*
dtype02A
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?
0model_15/fixed_adjacency_graph_convolution_5/addAddV2?model_15/fixed_adjacency_graph_convolution_5/Reshape_5:output:0Gmodel_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F22
0model_15/fixed_adjacency_graph_convolution_5/add?
model_15/reshape_30/ShapeShape4model_15/fixed_adjacency_graph_convolution_5/add:z:0*
T0*
_output_shapes
:2
model_15/reshape_30/Shape?
'model_15/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_30/strided_slice/stack?
)model_15/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_30/strided_slice/stack_1?
)model_15/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_30/strided_slice/stack_2?
!model_15/reshape_30/strided_sliceStridedSlice"model_15/reshape_30/Shape:output:00model_15/reshape_30/strided_slice/stack:output:02model_15/reshape_30/strided_slice/stack_1:output:02model_15/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_30/strided_slice?
#model_15/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_30/Reshape/shape/1?
#model_15/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model_15/reshape_30/Reshape/shape/2?
#model_15/reshape_30/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_15/reshape_30/Reshape/shape/3?
!model_15/reshape_30/Reshape/shapePack*model_15/reshape_30/strided_slice:output:0,model_15/reshape_30/Reshape/shape/1:output:0,model_15/reshape_30/Reshape/shape/2:output:0,model_15/reshape_30/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_30/Reshape/shape?
model_15/reshape_30/ReshapeReshape4model_15/fixed_adjacency_graph_convolution_5/add:z:0*model_15/reshape_30/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
model_15/reshape_30/Reshape?
!model_15/permute_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!model_15/permute_7/transpose/perm?
model_15/permute_7/transpose	Transpose$model_15/reshape_30/Reshape:output:0*model_15/permute_7/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
model_15/permute_7/transpose?
model_15/reshape_31/ShapeShape model_15/permute_7/transpose:y:0*
T0*
_output_shapes
:2
model_15/reshape_31/Shape?
'model_15/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/reshape_31/strided_slice/stack?
)model_15/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_31/strided_slice/stack_1?
)model_15/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_15/reshape_31/strided_slice/stack_2?
!model_15/reshape_31/strided_sliceStridedSlice"model_15/reshape_31/Shape:output:00model_15/reshape_31/strided_slice/stack:output:02model_15/reshape_31/strided_slice/stack_1:output:02model_15/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_15/reshape_31/strided_slice?
#model_15/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model_15/reshape_31/Reshape/shape/1?
#model_15/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2%
#model_15/reshape_31/Reshape/shape/2?
!model_15/reshape_31/Reshape/shapePack*model_15/reshape_31/strided_slice:output:0,model_15/reshape_31/Reshape/shape/1:output:0,model_15/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!model_15/reshape_31/Reshape/shape?
model_15/reshape_31/ReshapeReshape model_15/permute_7/transpose:y:0*model_15/reshape_31/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_15/reshape_31/Reshape?
model_15/lstm_5/ShapeShape$model_15/reshape_31/Reshape:output:0*
T0*
_output_shapes
:2
model_15/lstm_5/Shape?
#model_15/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_15/lstm_5/strided_slice/stack?
%model_15/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_15/lstm_5/strided_slice/stack_1?
%model_15/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_15/lstm_5/strided_slice/stack_2?
model_15/lstm_5/strided_sliceStridedSlicemodel_15/lstm_5/Shape:output:0,model_15/lstm_5/strided_slice/stack:output:0.model_15/lstm_5/strided_slice/stack_1:output:0.model_15/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_15/lstm_5/strided_slice|
model_15/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
model_15/lstm_5/zeros/mul/y?
model_15/lstm_5/zeros/mulMul&model_15/lstm_5/strided_slice:output:0$model_15/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros/mul
model_15/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_15/lstm_5/zeros/Less/y?
model_15/lstm_5/zeros/LessLessmodel_15/lstm_5/zeros/mul:z:0%model_15/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros/Less?
model_15/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2 
model_15/lstm_5/zeros/packed/1?
model_15/lstm_5/zeros/packedPack&model_15/lstm_5/strided_slice:output:0'model_15/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_15/lstm_5/zeros/packed
model_15/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/zeros/Const?
model_15/lstm_5/zerosFill%model_15/lstm_5/zeros/packed:output:0$model_15/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_15/lstm_5/zeros?
model_15/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
model_15/lstm_5/zeros_1/mul/y?
model_15/lstm_5/zeros_1/mulMul&model_15/lstm_5/strided_slice:output:0&model_15/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros_1/mul?
model_15/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
model_15/lstm_5/zeros_1/Less/y?
model_15/lstm_5/zeros_1/LessLessmodel_15/lstm_5/zeros_1/mul:z:0'model_15/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_15/lstm_5/zeros_1/Less?
 model_15/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 model_15/lstm_5/zeros_1/packed/1?
model_15/lstm_5/zeros_1/packedPack&model_15/lstm_5/strided_slice:output:0)model_15/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_15/lstm_5/zeros_1/packed?
model_15/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/zeros_1/Const?
model_15/lstm_5/zeros_1Fill'model_15/lstm_5/zeros_1/packed:output:0&model_15/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_15/lstm_5/zeros_1?
model_15/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_15/lstm_5/transpose/perm?
model_15/lstm_5/transpose	Transpose$model_15/reshape_31/Reshape:output:0'model_15/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
model_15/lstm_5/transpose
model_15/lstm_5/Shape_1Shapemodel_15/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
model_15/lstm_5/Shape_1?
%model_15/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_15/lstm_5/strided_slice_1/stack?
'model_15/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_1/stack_1?
'model_15/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_1/stack_2?
model_15/lstm_5/strided_slice_1StridedSlice model_15/lstm_5/Shape_1:output:0.model_15/lstm_5/strided_slice_1/stack:output:00model_15/lstm_5/strided_slice_1/stack_1:output:00model_15/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_15/lstm_5/strided_slice_1?
+model_15/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+model_15/lstm_5/TensorArrayV2/element_shape?
model_15/lstm_5/TensorArrayV2TensorListReserve4model_15/lstm_5/TensorArrayV2/element_shape:output:0(model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_15/lstm_5/TensorArrayV2?
Emodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2G
Emodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
7model_15/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_15/lstm_5/transpose:y:0Nmodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor?
%model_15/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_15/lstm_5/strided_slice_2/stack?
'model_15/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_2/stack_1?
'model_15/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_2/stack_2?
model_15/lstm_5/strided_slice_2StridedSlicemodel_15/lstm_5/transpose:y:0.model_15/lstm_5/strided_slice_2/stack:output:00model_15/lstm_5/strided_slice_2/stack_1:output:00model_15/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2!
model_15/lstm_5/strided_slice_2?
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype023
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
"model_15/lstm_5/lstm_cell_5/MatMulMatMul(model_15/lstm_5/strided_slice_2:output:09model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_15/lstm_5/lstm_cell_5/MatMul?
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype025
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
$model_15/lstm_5/lstm_cell_5/MatMul_1MatMulmodel_15/lstm_5/zeros:output:0;model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_15/lstm_5/lstm_cell_5/MatMul_1?
model_15/lstm_5/lstm_cell_5/addAddV2,model_15/lstm_5/lstm_cell_5/MatMul:product:0.model_15/lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2!
model_15/lstm_5/lstm_cell_5/add?
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
#model_15/lstm_5/lstm_cell_5/BiasAddBiasAdd#model_15/lstm_5/lstm_cell_5/add:z:0:model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_15/lstm_5/lstm_cell_5/BiasAdd?
!model_15/lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_15/lstm_5/lstm_cell_5/Const?
+model_15/lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_15/lstm_5/lstm_cell_5/split/split_dim?
!model_15/lstm_5/lstm_cell_5/splitSplit4model_15/lstm_5/lstm_cell_5/split/split_dim:output:0,model_15/lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2#
!model_15/lstm_5/lstm_cell_5/split?
#model_15/lstm_5/lstm_cell_5/SigmoidSigmoid*model_15/lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2%
#model_15/lstm_5/lstm_cell_5/Sigmoid?
%model_15/lstm_5/lstm_cell_5/Sigmoid_1Sigmoid*model_15/lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/lstm_cell_5/Sigmoid_1?
model_15/lstm_5/lstm_cell_5/mulMul)model_15/lstm_5/lstm_cell_5/Sigmoid_1:y:0 model_15/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????d2!
model_15/lstm_5/lstm_cell_5/mul?
 model_15/lstm_5/lstm_cell_5/ReluRelu*model_15/lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2"
 model_15/lstm_5/lstm_cell_5/Relu?
!model_15/lstm_5/lstm_cell_5/mul_1Mul'model_15/lstm_5/lstm_cell_5/Sigmoid:y:0.model_15/lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/mul_1?
!model_15/lstm_5/lstm_cell_5/add_1AddV2#model_15/lstm_5/lstm_cell_5/mul:z:0%model_15/lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/add_1?
%model_15/lstm_5/lstm_cell_5/Sigmoid_2Sigmoid*model_15/lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2'
%model_15/lstm_5/lstm_cell_5/Sigmoid_2?
"model_15/lstm_5/lstm_cell_5/Relu_1Relu%model_15/lstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2$
"model_15/lstm_5/lstm_cell_5/Relu_1?
!model_15/lstm_5/lstm_cell_5/mul_2Mul)model_15/lstm_5/lstm_cell_5/Sigmoid_2:y:00model_15/lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2#
!model_15/lstm_5/lstm_cell_5/mul_2?
-model_15/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2/
-model_15/lstm_5/TensorArrayV2_1/element_shape?
model_15/lstm_5/TensorArrayV2_1TensorListReserve6model_15/lstm_5/TensorArrayV2_1/element_shape:output:0(model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_15/lstm_5/TensorArrayV2_1n
model_15/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_15/lstm_5/time?
(model_15/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(model_15/lstm_5/while/maximum_iterations?
"model_15/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_15/lstm_5/while/loop_counter?
model_15/lstm_5/whileWhile+model_15/lstm_5/while/loop_counter:output:01model_15/lstm_5/while/maximum_iterations:output:0model_15/lstm_5/time:output:0(model_15/lstm_5/TensorArrayV2_1:handle:0model_15/lstm_5/zeros:output:0 model_15/lstm_5/zeros_1:output:0(model_15/lstm_5/strided_slice_1:output:0Gmodel_15/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0:model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource<model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource;model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!model_15_lstm_5_while_body_101852*-
cond%R#
!model_15_lstm_5_while_cond_101851*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
model_15/lstm_5/while?
@model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2B
@model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
2model_15/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStackmodel_15/lstm_5/while:output:3Imodel_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype024
2model_15/lstm_5/TensorArrayV2Stack/TensorListStack?
%model_15/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model_15/lstm_5/strided_slice_3/stack?
'model_15/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_15/lstm_5/strided_slice_3/stack_1?
'model_15/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_15/lstm_5/strided_slice_3/stack_2?
model_15/lstm_5/strided_slice_3StridedSlice;model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0.model_15/lstm_5/strided_slice_3/stack:output:00model_15/lstm_5/strided_slice_3/stack_1:output:00model_15/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2!
model_15/lstm_5/strided_slice_3?
 model_15/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_15/lstm_5/transpose_1/perm?
model_15/lstm_5/transpose_1	Transpose;model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0)model_15/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
model_15/lstm_5/transpose_1?
model_15/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_15/lstm_5/runtime?
model_15/dropout_13/IdentityIdentity(model_15/lstm_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2
model_15/dropout_13/Identity?
'model_15/dense_12/MatMul/ReadVariableOpReadVariableOp0model_15_dense_12_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02)
'model_15/dense_12/MatMul/ReadVariableOp?
model_15/dense_12/MatMulMatMul%model_15/dropout_13/Identity:output:0/model_15/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/MatMul?
(model_15/dense_12/BiasAdd/ReadVariableOpReadVariableOp1model_15_dense_12_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02*
(model_15/dense_12/BiasAdd/ReadVariableOp?
model_15/dense_12/BiasAddBiasAdd"model_15/dense_12/MatMul:product:00model_15/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/BiasAdd?
model_15/dense_12/SigmoidSigmoid"model_15/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
model_15/dense_12/Sigmoid?
IdentityIdentitymodel_15/dense_12/Sigmoid:y:0 ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp)^model_15/dense_12/BiasAdd/ReadVariableOp(^model_15/dense_12/MatMul/ReadVariableOp@^model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpH^model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpH^model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp3^model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2^model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp4^model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp^model_15/lstm_5/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2T
(model_15/dense_12/BiasAdd/ReadVariableOp(model_15/dense_12/BiasAdd/ReadVariableOp2R
'model_15/dense_12/MatMul/ReadVariableOp'model_15/dense_12/MatMul/ReadVariableOp2?
?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp2?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpGmodel_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp2?
Gmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpGmodel_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp2h
2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2f
1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp1model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp2j
3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp3model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp2.
model_15/lstm_5/whilemodel_15/lstm_5/while:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101233
input_22
dense_13_101208
dense_13_101210
model_15_101215
model_15_101217
model_15_101219
model_15_101221
model_15_101223
model_15_101225
model_15_101227
model_15_101229
identity?? dense_13/StatefulPartitionedCall? model_15/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_13_101208dense_13_101210*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1010782"
 dense_13/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011112
dropout_12/PartitionedCall?
reshape_28/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_28_layer_call_and_return_conditional_losses_1011372
reshape_28/PartitionedCall?
 model_15/StatefulPartitionedCallStatefulPartitionedCall#reshape_28/PartitionedCall:output:0model_15_101215model_15_101217model_15_101219model_15_101221model_15_101223model_15_101225model_15_101227model_15_101229*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1010252"
 model_15/StatefulPartitionedCall?
IdentityIdentity)model_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^model_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_15/StatefulPartitionedCall model_15/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
?g
?
)T-GCN-WX_model_15_lstm_5_while_body_99658N
Jt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_loop_counterT
Pt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_maximum_iterations.
*t_gcn_wx_model_15_lstm_5_while_placeholder0
,t_gcn_wx_model_15_lstm_5_while_placeholder_10
,t_gcn_wx_model_15_lstm_5_while_placeholder_20
,t_gcn_wx_model_15_lstm_5_while_placeholder_3M
It_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_strided_slice_1_0?
?t_gcn_wx_model_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0O
Kt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0Q
Mt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0P
Lt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0+
't_gcn_wx_model_15_lstm_5_while_identity-
)t_gcn_wx_model_15_lstm_5_while_identity_1-
)t_gcn_wx_model_15_lstm_5_while_identity_2-
)t_gcn_wx_model_15_lstm_5_while_identity_3-
)t_gcn_wx_model_15_lstm_5_while_identity_4-
)t_gcn_wx_model_15_lstm_5_while_identity_5K
Gt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_strided_slice_1?
?t_gcn_wx_model_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensorM
It_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceO
Kt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceN
Jt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??AT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?@T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?BT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
PT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2R
PT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
BT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?t_gcn_wx_model_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*t_gcn_wx_model_15_lstm_5_while_placeholderYT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02D
BT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
@T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpKt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02B
@T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
1T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMulMatMulIT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0HT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul?
BT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpMt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	d?*
dtype02D
BT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
3T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1MatMul,t_gcn_wx_model_15_lstm_5_while_placeholder_2JT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1?
.T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/addAddV2;T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul:product:0=T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????20
.T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add?
AT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpLt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02C
AT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
2T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAddBiasAdd2T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add:z:0IT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd?
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :22
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Const?
:T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split/split_dim?
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/splitSplitCT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split/split_dim:output:0;T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split22
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split?
2T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/SigmoidSigmoid9T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d24
2T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid?
4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid9T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d26
4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_1?
.T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mulMul8T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_1:y:0,t_gcn_wx_model_15_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????d20
.T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul?
/T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/ReluRelu9T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d21
/T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Relu?
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_1Mul6T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid:y:0=T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d22
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_1?
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add_1AddV22T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul:z:04T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d22
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add_1?
4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid9T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d26
4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_2?
1T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Relu_1Relu4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d23
1T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Relu_1?
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_2Mul8T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Sigmoid_2:y:0?T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d22
0T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_2?
CT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem,t_gcn_wx_model_15_lstm_5_while_placeholder_1*t_gcn_wx_model_15_lstm_5_while_placeholder4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02E
CT-GCN-WX/model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem?
$T-GCN-WX/model_15/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$T-GCN-WX/model_15/lstm_5/while/add/y?
"T-GCN-WX/model_15/lstm_5/while/addAddV2*t_gcn_wx_model_15_lstm_5_while_placeholder-T-GCN-WX/model_15/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_15/lstm_5/while/add?
&T-GCN-WX/model_15/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&T-GCN-WX/model_15/lstm_5/while/add_1/y?
$T-GCN-WX/model_15/lstm_5/while/add_1AddV2Jt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_loop_counter/T-GCN-WX/model_15/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2&
$T-GCN-WX/model_15/lstm_5/while/add_1?
'T-GCN-WX/model_15/lstm_5/while/IdentityIdentity(T-GCN-WX/model_15/lstm_5/while/add_1:z:0B^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2)
'T-GCN-WX/model_15/lstm_5/while/Identity?
)T-GCN-WX/model_15/lstm_5/while/Identity_1IdentityPt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_while_maximum_iterationsB^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)T-GCN-WX/model_15/lstm_5/while/Identity_1?
)T-GCN-WX/model_15/lstm_5/while/Identity_2Identity&T-GCN-WX/model_15/lstm_5/while/add:z:0B^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)T-GCN-WX/model_15/lstm_5/while/Identity_2?
)T-GCN-WX/model_15/lstm_5/while/Identity_3IdentityST-GCN-WX/model_15/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0B^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)T-GCN-WX/model_15/lstm_5/while/Identity_3?
)T-GCN-WX/model_15/lstm_5/while/Identity_4Identity4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/mul_2:z:0B^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2+
)T-GCN-WX/model_15/lstm_5/while/Identity_4?
)T-GCN-WX/model_15/lstm_5/while/Identity_5Identity4T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/add_1:z:0B^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpA^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpC^T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2+
)T-GCN-WX/model_15/lstm_5/while/Identity_5"[
't_gcn_wx_model_15_lstm_5_while_identity0T-GCN-WX/model_15/lstm_5/while/Identity:output:0"_
)t_gcn_wx_model_15_lstm_5_while_identity_12T-GCN-WX/model_15/lstm_5/while/Identity_1:output:0"_
)t_gcn_wx_model_15_lstm_5_while_identity_22T-GCN-WX/model_15/lstm_5/while/Identity_2:output:0"_
)t_gcn_wx_model_15_lstm_5_while_identity_32T-GCN-WX/model_15/lstm_5/while/Identity_3:output:0"_
)t_gcn_wx_model_15_lstm_5_while_identity_42T-GCN-WX/model_15/lstm_5/while/Identity_4:output:0"_
)t_gcn_wx_model_15_lstm_5_while_identity_52T-GCN-WX/model_15/lstm_5/while/Identity_5:output:0"?
Jt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resourceLt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
Kt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceMt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
It_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceKt_gcn_wx_model_15_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
Gt_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_strided_slice_1It_gcn_wx_model_15_lstm_5_while_t_gcn_wx_model_15_lstm_5_strided_slice_1_0"?
?t_gcn_wx_model_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor?t_gcn_wx_model_15_lstm_5_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_15_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????d:?????????d: : :::2?
AT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpAT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp2?
@T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp@T-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp2?
BT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpBT-GCN-WX/model_15/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101264

inputs
dense_13_101239
dense_13_101241
model_15_101246
model_15_101248
model_15_101250
model_15_101252
model_15_101254
model_15_101256
model_15_101258
model_15_101260
identity?? dense_13/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall? model_15/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_101239dense_13_101241*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1010782"
 dense_13/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011062$
"dropout_12/StatefulPartitionedCall?
reshape_28/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_28_layer_call_and_return_conditional_losses_1011372
reshape_28/PartitionedCall?
 model_15/StatefulPartitionedCallStatefulPartitionedCall#reshape_28/PartitionedCall:output:0model_15_101246model_15_101248model_15_101250model_15_101252model_15_101254model_15_101256model_15_101258model_15_101260*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1009742"
 model_15/StatefulPartitionedCall?
IdentityIdentity)model_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall!^model_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2D
 model_15/StatefulPartitionedCall model_15/StatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?(
?
D__inference_model_15_layer_call_and_return_conditional_losses_100911
input_23.
*fixed_adjacency_graph_convolution_5_100467.
*fixed_adjacency_graph_convolution_5_100469.
*fixed_adjacency_graph_convolution_5_100471
lstm_5_100846
lstm_5_100848
lstm_5_100850
dense_12_100905
dense_12_100907
identity?? dense_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinput_23(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDims?
reshape_29/PartitionedCallPartitionedCall$tf.expand_dims_7/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_29_layer_call_and_return_conditional_losses_1003932
reshape_29/PartitionedCall?
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_29/PartitionedCall:output:0*fixed_adjacency_graph_convolution_5_100467*fixed_adjacency_graph_convolution_5_100469*fixed_adjacency_graph_convolution_5_100471*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_1004542=
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?
reshape_30/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_30_layer_call_and_return_conditional_losses_1004882
reshape_30/PartitionedCall?
permute_7/PartitionedCallPartitionedCall#reshape_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_permute_7_layer_call_and_return_conditional_losses_997582
permute_7/PartitionedCall?
reshape_31/PartitionedCallPartitionedCall"permute_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_31_layer_call_and_return_conditional_losses_1005102
reshape_31/PartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_31/PartitionedCall:output:0lstm_5_100846lstm_5_100848lstm_5_100850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1006702 
lstm_5/StatefulPartitionedCall?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008652$
"dropout_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_12_100905dense_12_100907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1008942"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_23
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103037
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_102952*
condR
while_cond_102951*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
??
?
"__inference__traced_restore_103789
file_prefix$
 assignvariableop_dense_13_kernel$
 assignvariableop_1_dense_13_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rateA
=assignvariableop_7_fixed_adjacency_graph_convolution_5_kernel?
;assignvariableop_8_fixed_adjacency_graph_convolution_5_bias0
,assignvariableop_9_lstm_5_lstm_cell_5_kernel;
7assignvariableop_10_lstm_5_lstm_cell_5_recurrent_kernel/
+assignvariableop_11_lstm_5_lstm_cell_5_bias'
#assignvariableop_12_dense_12_kernel%
!assignvariableop_13_dense_12_bias=
9assignvariableop_14_fixed_adjacency_graph_convolution_5_a
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1.
*assignvariableop_19_adam_dense_13_kernel_m,
(assignvariableop_20_adam_dense_13_bias_mI
Eassignvariableop_21_adam_fixed_adjacency_graph_convolution_5_kernel_mG
Cassignvariableop_22_adam_fixed_adjacency_graph_convolution_5_bias_m8
4assignvariableop_23_adam_lstm_5_lstm_cell_5_kernel_mB
>assignvariableop_24_adam_lstm_5_lstm_cell_5_recurrent_kernel_m6
2assignvariableop_25_adam_lstm_5_lstm_cell_5_bias_m.
*assignvariableop_26_adam_dense_12_kernel_m,
(assignvariableop_27_adam_dense_12_bias_m.
*assignvariableop_28_adam_dense_13_kernel_v,
(assignvariableop_29_adam_dense_13_bias_vI
Eassignvariableop_30_adam_fixed_adjacency_graph_convolution_5_kernel_vG
Cassignvariableop_31_adam_fixed_adjacency_graph_convolution_5_bias_v8
4assignvariableop_32_adam_lstm_5_lstm_cell_5_kernel_vB
>assignvariableop_33_adam_lstm_5_lstm_cell_5_recurrent_kernel_v6
2assignvariableop_34_adam_lstm_5_lstm_cell_5_bias_v.
*assignvariableop_35_adam_dense_12_kernel_v,
(assignvariableop_36_adam_dense_12_bias_v
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_13_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_13_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp=assignvariableop_7_fixed_adjacency_graph_convolution_5_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp;assignvariableop_8_fixed_adjacency_graph_convolution_5_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_5_lstm_cell_5_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_5_lstm_cell_5_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_5_lstm_cell_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_fixed_adjacency_graph_convolution_5_aIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_13_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_13_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpEassignvariableop_21_adam_fixed_adjacency_graph_convolution_5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpCassignvariableop_22_adam_fixed_adjacency_graph_convolution_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_5_lstm_cell_5_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_5_lstm_cell_5_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_5_lstm_cell_5_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_12_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_12_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_13_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_13_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpEassignvariableop_30_adam_fixed_adjacency_graph_convolution_5_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpCassignvariableop_31_adam_fixed_adjacency_graph_convolution_5_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_5_lstm_cell_5_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_lstm_5_lstm_cell_5_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_lstm_5_lstm_cell_5_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_12_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_12_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
?
b
F__inference_reshape_31_layer_call_and_return_conditional_losses_100510

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
strided_slice/stack_2?
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
?????????2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
G
+__inference_reshape_31_layer_call_fn_102731

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_31_layer_call_and_return_conditional_losses_1005102
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
E
)__inference_permute_7_layer_call_fn_99764

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_permute_7_layer_call_and_return_conditional_losses_997582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_13_layer_call_fn_103409

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101317

inputs
dense_13_101292
dense_13_101294
model_15_101299
model_15_101301
model_15_101303
model_15_101305
model_15_101307
model_15_101309
model_15_101311
model_15_101313
identity?? dense_13/StatefulPartitionedCall? model_15/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_101292dense_13_101294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_1010782"
 dense_13/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011112
dropout_12/PartitionedCall?
reshape_28/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_28_layer_call_and_return_conditional_losses_1011372
reshape_28/PartitionedCall?
 model_15/StatefulPartitionedCallStatefulPartitionedCall#reshape_28/PartitionedCall:output:0model_15_101299model_15_101301model_15_101303model_15_101305model_15_101307model_15_101309model_15_101311model_15_101313*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_15_layer_call_and_return_conditional_losses_1010252"
 model_15/StatefulPartitionedCall?
IdentityIdentity)model_15/StatefulPartitionedCall:output:0!^dense_13/StatefulPartitionedCall!^model_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 model_15/StatefulPartitionedCall model_15/StatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
)__inference_T-GCN-WX_layer_call_fn_101287
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_1012642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
?
?
'__inference_lstm_5_layer_call_fn_103059
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1003652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?
G
+__inference_dropout_13_layer_call_fn_103414

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008702
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
'__inference_lstm_5_layer_call_fn_103376

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1006702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_103404

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_99837

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????d2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????d2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????d2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????d2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????d2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????d2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????d2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????d2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????F:?????????d:?????????d:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?
d
+__inference_dropout_12_layer_call_fn_102056

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?&
?
D__inference_model_15_layer_call_and_return_conditional_losses_100941
input_23.
*fixed_adjacency_graph_convolution_5_100917.
*fixed_adjacency_graph_convolution_5_100919.
*fixed_adjacency_graph_convolution_5_100921
lstm_5_100927
lstm_5_100929
lstm_5_100931
dense_12_100935
dense_12_100937
identity?? dense_12/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimsinput_23(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_7/ExpandDims?
reshape_29/PartitionedCallPartitionedCall$tf.expand_dims_7/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_29_layer_call_and_return_conditional_losses_1003932
reshape_29/PartitionedCall?
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_29/PartitionedCall:output:0*fixed_adjacency_graph_convolution_5_100917*fixed_adjacency_graph_convolution_5_100919*fixed_adjacency_graph_convolution_5_100921*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_1004542=
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall?
reshape_30/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_30_layer_call_and_return_conditional_losses_1004882
reshape_30/PartitionedCall?
permute_7/PartitionedCallPartitionedCall#reshape_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_permute_7_layer_call_and_return_conditional_losses_997582
permute_7/PartitionedCall?
reshape_31/PartitionedCallPartitionedCall"permute_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_31_layer_call_and_return_conditional_losses_1005102
reshape_31/PartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall#reshape_31/PartitionedCall:output:0lstm_5_100927lstm_5_100929lstm_5_100931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1008232 
lstm_5/StatefulPartitionedCall?
dropout_13/PartitionedCallPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_1008702
dropout_13/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_12_100935dense_12_100937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1008942"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall;fixed_adjacency_graph_convolution_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_23
?
?
D__inference_fixed_adjacency_graph_convolution_5_layer_call_fn_102694
features
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeaturesunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_1004542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
features
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103365

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
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
:?????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_103280*
condR
while_cond_103279*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?D
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_100233

inputs
lstm_cell_5_100151
lstm_cell_5_100153
lstm_cell_5_100155
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_100151lstm_cell_5_100153lstm_cell_5_100155*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_998372%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_100151lstm_cell_5_100153lstm_cell_5_100155*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_100164*
condR
while_cond_100163*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?Z
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_100823

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
zeros/packed/1?
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
:?????????d2
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
B :?2
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
zeros_1/packed/1?
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
:?????????d2	
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
:?????????F2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_100738*
condR
while_cond_100737*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
~
)__inference_dense_12_layer_call_fn_103434

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????F*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_1008942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_99751
input_227
3t_gcn_wx_dense_13_tensordot_readvariableop_resource5
1t_gcn_wx_dense_13_biasadd_readvariableop_resourceY
Ut_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resourceY
Ut_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resourceU
Qt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resourceG
Ct_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resourceI
Et_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resourceH
Dt_gcn_wx_model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource=
9t_gcn_wx_model_15_dense_12_matmul_readvariableop_resource>
:t_gcn_wx_model_15_dense_12_biasadd_readvariableop_resource
identity??(T-GCN-WX/dense_13/BiasAdd/ReadVariableOp?*T-GCN-WX/dense_13/Tensordot/ReadVariableOp?1T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp?0T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOp?HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?;T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?:T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?<T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?T-GCN-WX/model_15/lstm_5/while?
*T-GCN-WX/dense_13/Tensordot/ReadVariableOpReadVariableOp3t_gcn_wx_dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02,
*T-GCN-WX/dense_13/Tensordot/ReadVariableOp?
 T-GCN-WX/dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 T-GCN-WX/dense_13/Tensordot/axes?
 T-GCN-WX/dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 T-GCN-WX/dense_13/Tensordot/free~
!T-GCN-WX/dense_13/Tensordot/ShapeShapeinput_22*
T0*
_output_shapes
:2#
!T-GCN-WX/dense_13/Tensordot/Shape?
)T-GCN-WX/dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)T-GCN-WX/dense_13/Tensordot/GatherV2/axis?
$T-GCN-WX/dense_13/Tensordot/GatherV2GatherV2*T-GCN-WX/dense_13/Tensordot/Shape:output:0)T-GCN-WX/dense_13/Tensordot/free:output:02T-GCN-WX/dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$T-GCN-WX/dense_13/Tensordot/GatherV2?
+T-GCN-WX/dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+T-GCN-WX/dense_13/Tensordot/GatherV2_1/axis?
&T-GCN-WX/dense_13/Tensordot/GatherV2_1GatherV2*T-GCN-WX/dense_13/Tensordot/Shape:output:0)T-GCN-WX/dense_13/Tensordot/axes:output:04T-GCN-WX/dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&T-GCN-WX/dense_13/Tensordot/GatherV2_1?
!T-GCN-WX/dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!T-GCN-WX/dense_13/Tensordot/Const?
 T-GCN-WX/dense_13/Tensordot/ProdProd-T-GCN-WX/dense_13/Tensordot/GatherV2:output:0*T-GCN-WX/dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 T-GCN-WX/dense_13/Tensordot/Prod?
#T-GCN-WX/dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#T-GCN-WX/dense_13/Tensordot/Const_1?
"T-GCN-WX/dense_13/Tensordot/Prod_1Prod/T-GCN-WX/dense_13/Tensordot/GatherV2_1:output:0,T-GCN-WX/dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"T-GCN-WX/dense_13/Tensordot/Prod_1?
'T-GCN-WX/dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'T-GCN-WX/dense_13/Tensordot/concat/axis?
"T-GCN-WX/dense_13/Tensordot/concatConcatV2)T-GCN-WX/dense_13/Tensordot/free:output:0)T-GCN-WX/dense_13/Tensordot/axes:output:00T-GCN-WX/dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"T-GCN-WX/dense_13/Tensordot/concat?
!T-GCN-WX/dense_13/Tensordot/stackPack)T-GCN-WX/dense_13/Tensordot/Prod:output:0+T-GCN-WX/dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/dense_13/Tensordot/stack?
%T-GCN-WX/dense_13/Tensordot/transpose	Transposeinput_22+T-GCN-WX/dense_13/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2'
%T-GCN-WX/dense_13/Tensordot/transpose?
#T-GCN-WX/dense_13/Tensordot/ReshapeReshape)T-GCN-WX/dense_13/Tensordot/transpose:y:0*T-GCN-WX/dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2%
#T-GCN-WX/dense_13/Tensordot/Reshape?
"T-GCN-WX/dense_13/Tensordot/MatMulMatMul,T-GCN-WX/dense_13/Tensordot/Reshape:output:02T-GCN-WX/dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"T-GCN-WX/dense_13/Tensordot/MatMul?
#T-GCN-WX/dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#T-GCN-WX/dense_13/Tensordot/Const_2?
)T-GCN-WX/dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)T-GCN-WX/dense_13/Tensordot/concat_1/axis?
$T-GCN-WX/dense_13/Tensordot/concat_1ConcatV2-T-GCN-WX/dense_13/Tensordot/GatherV2:output:0,T-GCN-WX/dense_13/Tensordot/Const_2:output:02T-GCN-WX/dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$T-GCN-WX/dense_13/Tensordot/concat_1?
T-GCN-WX/dense_13/TensordotReshape,T-GCN-WX/dense_13/Tensordot/MatMul:product:0-T-GCN-WX/dense_13/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dense_13/Tensordot?
(T-GCN-WX/dense_13/BiasAdd/ReadVariableOpReadVariableOp1t_gcn_wx_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(T-GCN-WX/dense_13/BiasAdd/ReadVariableOp?
T-GCN-WX/dense_13/BiasAddBiasAdd$T-GCN-WX/dense_13/Tensordot:output:00T-GCN-WX/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dense_13/BiasAdd?
T-GCN-WX/dropout_12/IdentityIdentity"T-GCN-WX/dense_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dropout_12/Identity?
T-GCN-WX/reshape_28/ShapeShape%T-GCN-WX/dropout_12/Identity:output:0*
T0*
_output_shapes
:2
T-GCN-WX/reshape_28/Shape?
'T-GCN-WX/reshape_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'T-GCN-WX/reshape_28/strided_slice/stack?
)T-GCN-WX/reshape_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_28/strided_slice/stack_1?
)T-GCN-WX/reshape_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_28/strided_slice/stack_2?
!T-GCN-WX/reshape_28/strided_sliceStridedSlice"T-GCN-WX/reshape_28/Shape:output:00T-GCN-WX/reshape_28/strided_slice/stack:output:02T-GCN-WX/reshape_28/strided_slice/stack_1:output:02T-GCN-WX/reshape_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!T-GCN-WX/reshape_28/strided_slice?
#T-GCN-WX/reshape_28/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#T-GCN-WX/reshape_28/Reshape/shape/1?
#T-GCN-WX/reshape_28/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#T-GCN-WX/reshape_28/Reshape/shape/2?
!T-GCN-WX/reshape_28/Reshape/shapePack*T-GCN-WX/reshape_28/strided_slice:output:0,T-GCN-WX/reshape_28/Reshape/shape/1:output:0,T-GCN-WX/reshape_28/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/reshape_28/Reshape/shape?
T-GCN-WX/reshape_28/ReshapeReshape%T-GCN-WX/dropout_12/Identity:output:0*T-GCN-WX/reshape_28/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
T-GCN-WX/reshape_28/Reshape?
1T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims/dim?
-T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims
ExpandDims$T-GCN-WX/reshape_28/Reshape:output:0:T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2/
-T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims?
"T-GCN-WX/model_15/reshape_29/ShapeShape6T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims:output:0*
T0*
_output_shapes
:2$
"T-GCN-WX/model_15/reshape_29/Shape?
0T-GCN-WX/model_15/reshape_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0T-GCN-WX/model_15/reshape_29/strided_slice/stack?
2T-GCN-WX/model_15/reshape_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_29/strided_slice/stack_1?
2T-GCN-WX/model_15/reshape_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_29/strided_slice/stack_2?
*T-GCN-WX/model_15/reshape_29/strided_sliceStridedSlice+T-GCN-WX/model_15/reshape_29/Shape:output:09T-GCN-WX/model_15/reshape_29/strided_slice/stack:output:0;T-GCN-WX/model_15/reshape_29/strided_slice/stack_1:output:0;T-GCN-WX/model_15/reshape_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*T-GCN-WX/model_15/reshape_29/strided_slice?
,T-GCN-WX/model_15/reshape_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2.
,T-GCN-WX/model_15/reshape_29/Reshape/shape/1?
,T-GCN-WX/model_15/reshape_29/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,T-GCN-WX/model_15/reshape_29/Reshape/shape/2?
*T-GCN-WX/model_15/reshape_29/Reshape/shapePack3T-GCN-WX/model_15/reshape_29/strided_slice:output:05T-GCN-WX/model_15/reshape_29/Reshape/shape/1:output:05T-GCN-WX/model_15/reshape_29/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2,
*T-GCN-WX/model_15/reshape_29/Reshape/shape?
$T-GCN-WX/model_15/reshape_29/ReshapeReshape6T-GCN-WX/model_15/tf.expand_dims_7/ExpandDims:output:03T-GCN-WX/model_15/reshape_29/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2&
$T-GCN-WX/model_15/reshape_29/Reshape?
DT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2F
DT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose/perm?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose	Transpose-T-GCN-WX/model_15/reshape_29/Reshape:output:0MT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose?
;T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/ShapeShapeCT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose:y:0*
T0*
_output_shapes
:2=
;T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstackUnpackDT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape:output:0*
T0*
_output_shapes
: : : *	
num2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack?
LT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOpReadVariableOpUt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02N
LT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_1/ReadVariableOp?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_1?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_1UnpackFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_1?
CT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2E
CT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape/shape?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/ReshapeReshapeCT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose:y:0LT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape?
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpReadVariableOpUt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02R
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp?
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2H
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/perm?
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1	TransposeXT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp:value:0OT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2C
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1?
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2G
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1ReshapeET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1:y:0NT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1?
<T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMulMatMulFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape:output:0HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2>
<T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMul?
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2I
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1?
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2I
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2?
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shapePackFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack:output:0PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/1:output:0PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2G
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2ReshapeFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMul:product:0NT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2?
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2H
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2/perm?
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2	TransposeHT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_2:output:0OT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2C
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_2ShapeET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0*
T0*
_output_shapes
:2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_2?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_2UnpackFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_2?
LT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOpReadVariableOpUt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02N
LT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_3/ReadVariableOp?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2?
=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_3?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_3UnpackFT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_3?
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3ReshapeET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_2:y:0NT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3?
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpReadVariableOpUt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02R
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp?
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2H
FT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/perm?
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3	TransposeXT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp:value:0OT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/perm:output:0*
T0*
_output_shapes

:2C
AT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3?
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2G
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4ReshapeET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3:y:0NT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4/shape:output:0*
T0*
_output_shapes

:2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4?
>T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMul_1MatMulHT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_3:output:0HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2@
>T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMul_1?
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2I
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1?
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2I
GT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2?
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shapePackHT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/unstack_2:output:0PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/1:output:0PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2G
ET-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape?
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5ReshapeHT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/MatMul_1:product:0NT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2A
?T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5?
HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpReadVariableOpQt_gcn_wx_model_15_fixed_adjacency_graph_convolution_5_add_readvariableop_resource*
_output_shapes

:F*
dtype02J
HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp?
9T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/addAddV2HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/Reshape_5:output:0PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2;
9T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add?
"T-GCN-WX/model_15/reshape_30/ShapeShape=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add:z:0*
T0*
_output_shapes
:2$
"T-GCN-WX/model_15/reshape_30/Shape?
0T-GCN-WX/model_15/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0T-GCN-WX/model_15/reshape_30/strided_slice/stack?
2T-GCN-WX/model_15/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_30/strided_slice/stack_1?
2T-GCN-WX/model_15/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_30/strided_slice/stack_2?
*T-GCN-WX/model_15/reshape_30/strided_sliceStridedSlice+T-GCN-WX/model_15/reshape_30/Shape:output:09T-GCN-WX/model_15/reshape_30/strided_slice/stack:output:0;T-GCN-WX/model_15/reshape_30/strided_slice/stack_1:output:0;T-GCN-WX/model_15/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*T-GCN-WX/model_15/reshape_30/strided_slice?
,T-GCN-WX/model_15/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2.
,T-GCN-WX/model_15/reshape_30/Reshape/shape/1?
,T-GCN-WX/model_15/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,T-GCN-WX/model_15/reshape_30/Reshape/shape/2?
,T-GCN-WX/model_15/reshape_30/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,T-GCN-WX/model_15/reshape_30/Reshape/shape/3?
*T-GCN-WX/model_15/reshape_30/Reshape/shapePack3T-GCN-WX/model_15/reshape_30/strided_slice:output:05T-GCN-WX/model_15/reshape_30/Reshape/shape/1:output:05T-GCN-WX/model_15/reshape_30/Reshape/shape/2:output:05T-GCN-WX/model_15/reshape_30/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*T-GCN-WX/model_15/reshape_30/Reshape/shape?
$T-GCN-WX/model_15/reshape_30/ReshapeReshape=T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add:z:03T-GCN-WX/model_15/reshape_30/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2&
$T-GCN-WX/model_15/reshape_30/Reshape?
*T-GCN-WX/model_15/permute_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*T-GCN-WX/model_15/permute_7/transpose/perm?
%T-GCN-WX/model_15/permute_7/transpose	Transpose-T-GCN-WX/model_15/reshape_30/Reshape:output:03T-GCN-WX/model_15/permute_7/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2'
%T-GCN-WX/model_15/permute_7/transpose?
"T-GCN-WX/model_15/reshape_31/ShapeShape)T-GCN-WX/model_15/permute_7/transpose:y:0*
T0*
_output_shapes
:2$
"T-GCN-WX/model_15/reshape_31/Shape?
0T-GCN-WX/model_15/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0T-GCN-WX/model_15/reshape_31/strided_slice/stack?
2T-GCN-WX/model_15/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_31/strided_slice/stack_1?
2T-GCN-WX/model_15/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2T-GCN-WX/model_15/reshape_31/strided_slice/stack_2?
*T-GCN-WX/model_15/reshape_31/strided_sliceStridedSlice+T-GCN-WX/model_15/reshape_31/Shape:output:09T-GCN-WX/model_15/reshape_31/strided_slice/stack:output:0;T-GCN-WX/model_15/reshape_31/strided_slice/stack_1:output:0;T-GCN-WX/model_15/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*T-GCN-WX/model_15/reshape_31/strided_slice?
,T-GCN-WX/model_15/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,T-GCN-WX/model_15/reshape_31/Reshape/shape/1?
,T-GCN-WX/model_15/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2.
,T-GCN-WX/model_15/reshape_31/Reshape/shape/2?
*T-GCN-WX/model_15/reshape_31/Reshape/shapePack3T-GCN-WX/model_15/reshape_31/strided_slice:output:05T-GCN-WX/model_15/reshape_31/Reshape/shape/1:output:05T-GCN-WX/model_15/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2,
*T-GCN-WX/model_15/reshape_31/Reshape/shape?
$T-GCN-WX/model_15/reshape_31/ReshapeReshape)T-GCN-WX/model_15/permute_7/transpose:y:03T-GCN-WX/model_15/reshape_31/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2&
$T-GCN-WX/model_15/reshape_31/Reshape?
T-GCN-WX/model_15/lstm_5/ShapeShape-T-GCN-WX/model_15/reshape_31/Reshape:output:0*
T0*
_output_shapes
:2 
T-GCN-WX/model_15/lstm_5/Shape?
,T-GCN-WX/model_15/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,T-GCN-WX/model_15/lstm_5/strided_slice/stack?
.T-GCN-WX/model_15/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.T-GCN-WX/model_15/lstm_5/strided_slice/stack_1?
.T-GCN-WX/model_15/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.T-GCN-WX/model_15/lstm_5/strided_slice/stack_2?
&T-GCN-WX/model_15/lstm_5/strided_sliceStridedSlice'T-GCN-WX/model_15/lstm_5/Shape:output:05T-GCN-WX/model_15/lstm_5/strided_slice/stack:output:07T-GCN-WX/model_15/lstm_5/strided_slice/stack_1:output:07T-GCN-WX/model_15/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&T-GCN-WX/model_15/lstm_5/strided_slice?
$T-GCN-WX/model_15/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2&
$T-GCN-WX/model_15/lstm_5/zeros/mul/y?
"T-GCN-WX/model_15/lstm_5/zeros/mulMul/T-GCN-WX/model_15/lstm_5/strided_slice:output:0-T-GCN-WX/model_15/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_15/lstm_5/zeros/mul?
%T-GCN-WX/model_15/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%T-GCN-WX/model_15/lstm_5/zeros/Less/y?
#T-GCN-WX/model_15/lstm_5/zeros/LessLess&T-GCN-WX/model_15/lstm_5/zeros/mul:z:0.T-GCN-WX/model_15/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_15/lstm_5/zeros/Less?
'T-GCN-WX/model_15/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2)
'T-GCN-WX/model_15/lstm_5/zeros/packed/1?
%T-GCN-WX/model_15/lstm_5/zeros/packedPack/T-GCN-WX/model_15/lstm_5/strided_slice:output:00T-GCN-WX/model_15/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%T-GCN-WX/model_15/lstm_5/zeros/packed?
$T-GCN-WX/model_15/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$T-GCN-WX/model_15/lstm_5/zeros/Const?
T-GCN-WX/model_15/lstm_5/zerosFill.T-GCN-WX/model_15/lstm_5/zeros/packed:output:0-T-GCN-WX/model_15/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d2 
T-GCN-WX/model_15/lstm_5/zeros?
&T-GCN-WX/model_15/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2(
&T-GCN-WX/model_15/lstm_5/zeros_1/mul/y?
$T-GCN-WX/model_15/lstm_5/zeros_1/mulMul/T-GCN-WX/model_15/lstm_5/strided_slice:output:0/T-GCN-WX/model_15/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2&
$T-GCN-WX/model_15/lstm_5/zeros_1/mul?
'T-GCN-WX/model_15/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'T-GCN-WX/model_15/lstm_5/zeros_1/Less/y?
%T-GCN-WX/model_15/lstm_5/zeros_1/LessLess(T-GCN-WX/model_15/lstm_5/zeros_1/mul:z:00T-GCN-WX/model_15/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2'
%T-GCN-WX/model_15/lstm_5/zeros_1/Less?
)T-GCN-WX/model_15/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2+
)T-GCN-WX/model_15/lstm_5/zeros_1/packed/1?
'T-GCN-WX/model_15/lstm_5/zeros_1/packedPack/T-GCN-WX/model_15/lstm_5/strided_slice:output:02T-GCN-WX/model_15/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'T-GCN-WX/model_15/lstm_5/zeros_1/packed?
&T-GCN-WX/model_15/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&T-GCN-WX/model_15/lstm_5/zeros_1/Const?
 T-GCN-WX/model_15/lstm_5/zeros_1Fill0T-GCN-WX/model_15/lstm_5/zeros_1/packed:output:0/T-GCN-WX/model_15/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????d2"
 T-GCN-WX/model_15/lstm_5/zeros_1?
'T-GCN-WX/model_15/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'T-GCN-WX/model_15/lstm_5/transpose/perm?
"T-GCN-WX/model_15/lstm_5/transpose	Transpose-T-GCN-WX/model_15/reshape_31/Reshape:output:00T-GCN-WX/model_15/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2$
"T-GCN-WX/model_15/lstm_5/transpose?
 T-GCN-WX/model_15/lstm_5/Shape_1Shape&T-GCN-WX/model_15/lstm_5/transpose:y:0*
T0*
_output_shapes
:2"
 T-GCN-WX/model_15/lstm_5/Shape_1?
.T-GCN-WX/model_15/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.T-GCN-WX/model_15/lstm_5/strided_slice_1/stack?
0T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_1?
0T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_2?
(T-GCN-WX/model_15/lstm_5/strided_slice_1StridedSlice)T-GCN-WX/model_15/lstm_5/Shape_1:output:07T-GCN-WX/model_15/lstm_5/strided_slice_1/stack:output:09T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_1:output:09T-GCN-WX/model_15/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(T-GCN-WX/model_15/lstm_5/strided_slice_1?
4T-GCN-WX/model_15/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4T-GCN-WX/model_15/lstm_5/TensorArrayV2/element_shape?
&T-GCN-WX/model_15/lstm_5/TensorArrayV2TensorListReserve=T-GCN-WX/model_15/lstm_5/TensorArrayV2/element_shape:output:01T-GCN-WX/model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&T-GCN-WX/model_15/lstm_5/TensorArrayV2?
NT-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2P
NT-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
@T-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&T-GCN-WX/model_15/lstm_5/transpose:y:0WT-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@T-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor?
.T-GCN-WX/model_15/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.T-GCN-WX/model_15/lstm_5/strided_slice_2/stack?
0T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_1?
0T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_2?
(T-GCN-WX/model_15/lstm_5/strided_slice_2StridedSlice&T-GCN-WX/model_15/lstm_5/transpose:y:07T-GCN-WX/model_15/lstm_5/strided_slice_2/stack:output:09T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_1:output:09T-GCN-WX/model_15/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2*
(T-GCN-WX/model_15/lstm_5/strided_slice_2?
:T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpCt_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02<
:T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
+T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMulMatMul1T-GCN-WX/model_15/lstm_5/strided_slice_2:output:0BT-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul?
<T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpEt_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	d?*
dtype02>
<T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
-T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1MatMul'T-GCN-WX/model_15/lstm_5/zeros:output:0DT-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1?
(T-GCN-WX/model_15/lstm_5/lstm_cell_5/addAddV25T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul:product:07T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(T-GCN-WX/model_15/lstm_5/lstm_cell_5/add?
;T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpDt_gcn_wx_model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
,T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAddBiasAdd,T-GCN-WX/model_15/lstm_5/lstm_cell_5/add:z:0CT-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd?
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/Const?
4T-GCN-WX/model_15/lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4T-GCN-WX/model_15/lstm_5/lstm_cell_5/split/split_dim?
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/splitSplit=T-GCN-WX/model_15/lstm_5/lstm_cell_5/split/split_dim:output:05T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????d:?????????d:?????????d:?????????d*
	num_split2,
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/split?
,T-GCN-WX/model_15/lstm_5/lstm_cell_5/SigmoidSigmoid3T-GCN-WX/model_15/lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????d2.
,T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid?
.T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_1Sigmoid3T-GCN-WX/model_15/lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????d20
.T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_1?
(T-GCN-WX/model_15/lstm_5/lstm_cell_5/mulMul2T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_1:y:0)T-GCN-WX/model_15/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????d2*
(T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul?
)T-GCN-WX/model_15/lstm_5/lstm_cell_5/ReluRelu3T-GCN-WX/model_15/lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????d2+
)T-GCN-WX/model_15/lstm_5/lstm_cell_5/Relu?
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul_1Mul0T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid:y:07T-GCN-WX/model_15/lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????d2,
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul_1?
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/add_1AddV2,T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul:z:0.T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????d2,
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/add_1?
.T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_2Sigmoid3T-GCN-WX/model_15/lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????d20
.T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_2?
+T-GCN-WX/model_15/lstm_5/lstm_cell_5/Relu_1Relu.T-GCN-WX/model_15/lstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????d2-
+T-GCN-WX/model_15/lstm_5/lstm_cell_5/Relu_1?
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul_2Mul2T-GCN-WX/model_15/lstm_5/lstm_cell_5/Sigmoid_2:y:09T-GCN-WX/model_15/lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????d2,
*T-GCN-WX/model_15/lstm_5/lstm_cell_5/mul_2?
6T-GCN-WX/model_15/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   28
6T-GCN-WX/model_15/lstm_5/TensorArrayV2_1/element_shape?
(T-GCN-WX/model_15/lstm_5/TensorArrayV2_1TensorListReserve?T-GCN-WX/model_15/lstm_5/TensorArrayV2_1/element_shape:output:01T-GCN-WX/model_15/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(T-GCN-WX/model_15/lstm_5/TensorArrayV2_1?
T-GCN-WX/model_15/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
T-GCN-WX/model_15/lstm_5/time?
1T-GCN-WX/model_15/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1T-GCN-WX/model_15/lstm_5/while/maximum_iterations?
+T-GCN-WX/model_15/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2-
+T-GCN-WX/model_15/lstm_5/while/loop_counter?
T-GCN-WX/model_15/lstm_5/whileWhile4T-GCN-WX/model_15/lstm_5/while/loop_counter:output:0:T-GCN-WX/model_15/lstm_5/while/maximum_iterations:output:0&T-GCN-WX/model_15/lstm_5/time:output:01T-GCN-WX/model_15/lstm_5/TensorArrayV2_1:handle:0'T-GCN-WX/model_15/lstm_5/zeros:output:0)T-GCN-WX/model_15/lstm_5/zeros_1:output:01T-GCN-WX/model_15/lstm_5/strided_slice_1:output:0PT-GCN-WX/model_15/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ct_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_readvariableop_resourceEt_gcn_wx_model_15_lstm_5_lstm_cell_5_matmul_1_readvariableop_resourceDt_gcn_wx_model_15_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*5
body-R+
)T-GCN-WX_model_15_lstm_5_while_body_99658*5
cond-R+
)T-GCN-WX_model_15_lstm_5_while_cond_99657*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations 2 
T-GCN-WX/model_15/lstm_5/while?
IT-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2K
IT-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
;T-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack'T-GCN-WX/model_15/lstm_5/while:output:3RT-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype02=
;T-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack?
.T-GCN-WX/model_15/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????20
.T-GCN-WX/model_15/lstm_5/strided_slice_3/stack?
0T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_1?
0T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_2?
(T-GCN-WX/model_15/lstm_5/strided_slice_3StridedSliceDT-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:07T-GCN-WX/model_15/lstm_5/strided_slice_3/stack:output:09T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_1:output:09T-GCN-WX/model_15/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2*
(T-GCN-WX/model_15/lstm_5/strided_slice_3?
)T-GCN-WX/model_15/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)T-GCN-WX/model_15/lstm_5/transpose_1/perm?
$T-GCN-WX/model_15/lstm_5/transpose_1	TransposeDT-GCN-WX/model_15/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02T-GCN-WX/model_15/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d2&
$T-GCN-WX/model_15/lstm_5/transpose_1?
 T-GCN-WX/model_15/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2"
 T-GCN-WX/model_15/lstm_5/runtime?
%T-GCN-WX/model_15/dropout_13/IdentityIdentity1T-GCN-WX/model_15/lstm_5/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d2'
%T-GCN-WX/model_15/dropout_13/Identity?
0T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOpReadVariableOp9t_gcn_wx_model_15_dense_12_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype022
0T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOp?
!T-GCN-WX/model_15/dense_12/MatMulMatMul.T-GCN-WX/model_15/dropout_13/Identity:output:08T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2#
!T-GCN-WX/model_15/dense_12/MatMul?
1T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOpReadVariableOp:t_gcn_wx_model_15_dense_12_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype023
1T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp?
"T-GCN-WX/model_15/dense_12/BiasAddBiasAdd+T-GCN-WX/model_15/dense_12/MatMul:product:09T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2$
"T-GCN-WX/model_15/dense_12/BiasAdd?
"T-GCN-WX/model_15/dense_12/SigmoidSigmoid+T-GCN-WX/model_15/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2$
"T-GCN-WX/model_15/dense_12/Sigmoid?
IdentityIdentity&T-GCN-WX/model_15/dense_12/Sigmoid:y:0)^T-GCN-WX/dense_13/BiasAdd/ReadVariableOp+^T-GCN-WX/dense_13/Tensordot/ReadVariableOp2^T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp1^T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOpI^T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpQ^T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpQ^T-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp<^T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp;^T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp=^T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp^T-GCN-WX/model_15/lstm_5/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2T
(T-GCN-WX/dense_13/BiasAdd/ReadVariableOp(T-GCN-WX/dense_13/BiasAdd/ReadVariableOp2X
*T-GCN-WX/dense_13/Tensordot/ReadVariableOp*T-GCN-WX/dense_13/Tensordot/ReadVariableOp2f
1T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp1T-GCN-WX/model_15/dense_12/BiasAdd/ReadVariableOp2d
0T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOp0T-GCN-WX/model_15/dense_12/MatMul/ReadVariableOp2?
HT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOpHT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/add/ReadVariableOp2?
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOpPT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_1/ReadVariableOp2?
PT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOpPT-GCN-WX/model_15/fixed_adjacency_graph_convolution_5/transpose_3/ReadVariableOp2z
;T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp;T-GCN-WX/model_15/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp2x
:T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp:T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul/ReadVariableOp2|
<T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp<T-GCN-WX/model_15/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp2@
T-GCN-WX/model_15/lstm_5/whileT-GCN-WX/model_15/lstm_5/while:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_22
?
G
+__inference_dropout_12_layer_call_fn_102061

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_1011112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_229
serving_default_input_22:0?????????F<
model_150
StatefulPartitionedCall:0?????????Ftensorflow/serving/predict:??
?X
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?U
_tf_keras_network?U{"class_name": "Functional", "name": "T-GCN-WX", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_12", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_28", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_28", "inbound_nodes": [[["dropout_12", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["input_23", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_29", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_29", "inbound_nodes": [[["tf.expand_dims_7", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_5", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_5", "inbound_nodes": [[["reshape_29", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_30", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_30", "inbound_nodes": [[["fixed_adjacency_graph_convolution_5", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_7", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_7", "inbound_nodes": [[["reshape_30", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_31", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_31", "inbound_nodes": [[["permute_7", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_5", "inbound_nodes": [[["reshape_31", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_12", 0, 0]]}, "name": "model_15", "inbound_nodes": [[["reshape_28", 0, 0, {}]]]}], "input_layers": [["input_22", 0, 0]], "output_layers": [["model_15", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_12", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_28", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_28", "inbound_nodes": [[["dropout_12", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["input_23", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_29", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_29", "inbound_nodes": [[["tf.expand_dims_7", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_5", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_5", "inbound_nodes": [[["reshape_29", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_30", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_30", "inbound_nodes": [[["fixed_adjacency_graph_convolution_5", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_7", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_7", "inbound_nodes": [[["reshape_30", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_31", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_31", "inbound_nodes": [[["permute_7", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_5", "inbound_nodes": [[["reshape_31", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_12", 0, 0]]}, "name": "model_15", "inbound_nodes": [[["reshape_28", 0, 0, {}]]]}], "input_layers": [["input_22", 0, 0]], "output_layers": [["model_15", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_22", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_28", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
??
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
 layer-6
!layer_with_weights-1
!layer-7
"layer-8
#layer_with_weights-2
#layer-9
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?<
_tf_keras_network?<{"class_name": "Functional", "name": "model_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["input_23", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_29", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_29", "inbound_nodes": [[["tf.expand_dims_7", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_5", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_5", "inbound_nodes": [[["reshape_29", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_30", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_30", "inbound_nodes": [[["fixed_adjacency_graph_convolution_5", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_7", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_7", "inbound_nodes": [[["reshape_30", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_31", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_31", "inbound_nodes": [[["permute_7", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_5", "inbound_nodes": [[["reshape_31", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_12", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}, "name": "input_23", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["input_23", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_29", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_29", "inbound_nodes": [[["tf.expand_dims_7", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_5", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_5", "inbound_nodes": [[["reshape_29", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_30", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_30", "inbound_nodes": [[["fixed_adjacency_graph_convolution_5", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_7", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_7", "inbound_nodes": [[["reshape_30", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_31", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_31", "inbound_nodes": [[["permute_7", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_5", "inbound_nodes": [[["reshape_31", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["lstm_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_23", 0, 0]], "output_layers": [["dense_12", 0, 0]]}}}
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratem?m?-m?.m?/m?0m?1m?2m?3m?v?v?-v?.v?/v?0v?1v?2v?3v?"
	optimizer
_
0
1
-2
.3
/4
05
16
27
38"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
-2
.3
44
/5
06
17
28
39"
trackable_list_wrapper
?
5layer_metrics

6layers
trainable_variables
7metrics
8layer_regularization_losses
9non_trainable_variables
regularization_losses
		variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
!:2dense_13/kernel
:2dense_13/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
:layer_metrics

;layers
trainable_variables
<metrics
=layer_regularization_losses
>non_trainable_variables
regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics

@layers
trainable_variables
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_metrics

Elayers
trainable_variables
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_23", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_23"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_29", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
?
4A

-kernel
.bias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "FixedAdjacencyGraphConvolution", "name": "fixed_adjacency_graph_convolution_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_adjacency_graph_convolution_5", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}}
?
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_30", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}}
?
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Permute", "name": "permute_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "permute_7", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_31", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}}
?
^cell
_
state_spec
`trainable_variables
aregularization_losses
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 70]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 70]}}
?
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

2kernel
3bias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Q
-0
.1
/2
03
14
25
36"
trackable_list_wrapper
 "
trackable_list_wrapper
X
-0
.1
42
/3
04
15
26
37"
trackable_list_wrapper
?
llayer_metrics

mlayers
$trainable_variables
nmetrics
olayer_regularization_losses
pnon_trainable_variables
%regularization_losses
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<::2*fixed_adjacency_graph_convolution_5/kernel
::8F2(fixed_adjacency_graph_convolution_5/bias
,:*	F?2lstm_5/lstm_cell_5/kernel
6:4	d?2#lstm_5/lstm_cell_5/recurrent_kernel
&:$?2lstm_5/lstm_cell_5/bias
!:dF2dense_12/kernel
:F2dense_12/bias
5:3FF2%fixed_adjacency_graph_convolution_5/A
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
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
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
slayer_metrics

tlayers
Jtrainable_variables
umetrics
vlayer_regularization_losses
wnon_trainable_variables
Kregularization_losses
L	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
-0
.1
42"
trackable_list_wrapper
?
xlayer_metrics

ylayers
Ntrainable_variables
zmetrics
{layer_regularization_losses
|non_trainable_variables
Oregularization_losses
P	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}layer_metrics

~layers
Rtrainable_variables
metrics
 ?layer_regularization_losses
?non_trainable_variables
Sregularization_losses
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
Vtrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
Wregularization_losses
X	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
Ztrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
[regularization_losses
\	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

/kernel
0recurrent_kernel
1bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_5", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
?
?layer_metrics
?layers
`trainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?states
aregularization_losses
b	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
dtrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
eregularization_losses
f	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?layer_metrics
?layers
htrainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
iregularization_losses
j	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
 6
!7
"8
#9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
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
'
40"
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
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
?
?layer_metrics
?layers
?trainable_variables
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
^0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
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
&:$2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
A:?21Adam/fixed_adjacency_graph_convolution_5/kernel/m
?:=F2/Adam/fixed_adjacency_graph_convolution_5/bias/m
1:/	F?2 Adam/lstm_5/lstm_cell_5/kernel/m
;:9	d?2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
+:)?2Adam/lstm_5/lstm_cell_5/bias/m
&:$dF2Adam/dense_12/kernel/m
 :F2Adam/dense_12/bias/m
&:$2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
A:?21Adam/fixed_adjacency_graph_convolution_5/kernel/v
?:=F2/Adam/fixed_adjacency_graph_convolution_5/bias/v
1:/	F?2 Adam/lstm_5/lstm_cell_5/kernel/v
;:9	d?2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
+:)?2Adam/lstm_5/lstm_cell_5/bias/v
&:$dF2Adam/dense_12/kernel/v
 :F2Adam/dense_12/bias/v
?2?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101945
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101205
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101233
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101667?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_T-GCN-WX_layer_call_fn_101970
)__inference_T-GCN-WX_layer_call_fn_101340
)__inference_T-GCN-WX_layer_call_fn_101995
)__inference_T-GCN-WX_layer_call_fn_101287?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_99751?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_22?????????F
?2?
D__inference_dense_13_layer_call_and_return_conditional_losses_102025?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_13_layer_call_fn_102034?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_12_layer_call_and_return_conditional_losses_102046
F__inference_dropout_12_layer_call_and_return_conditional_losses_102051?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_12_layer_call_fn_102061
+__inference_dropout_12_layer_call_fn_102056?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_reshape_28_layer_call_and_return_conditional_losses_102074?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_28_layer_call_fn_102079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_model_15_layer_call_and_return_conditional_losses_100941
D__inference_model_15_layer_call_and_return_conditional_losses_100911
D__inference_model_15_layer_call_and_return_conditional_losses_102328
D__inference_model_15_layer_call_and_return_conditional_losses_102570?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_15_layer_call_fn_102612
)__inference_model_15_layer_call_fn_101044
)__inference_model_15_layer_call_fn_100993
)__inference_model_15_layer_call_fn_102591?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_101375input_22"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_29_layer_call_and_return_conditional_losses_102625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_29_layer_call_fn_102630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_102683?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_fixed_adjacency_graph_convolution_5_layer_call_fn_102694?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_30_layer_call_and_return_conditional_losses_102708?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_30_layer_call_fn_102713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_permute_7_layer_call_and_return_conditional_losses_99758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_permute_7_layer_call_fn_99764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_reshape_31_layer_call_and_return_conditional_losses_102726?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_31_layer_call_fn_102731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103037
B__inference_lstm_5_layer_call_and_return_conditional_losses_103212
B__inference_lstm_5_layer_call_and_return_conditional_losses_102884
B__inference_lstm_5_layer_call_and_return_conditional_losses_103365?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_5_layer_call_fn_103048
'__inference_lstm_5_layer_call_fn_103387
'__inference_lstm_5_layer_call_fn_103059
'__inference_lstm_5_layer_call_fn_103376?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_13_layer_call_and_return_conditional_losses_103404
F__inference_dropout_13_layer_call_and_return_conditional_losses_103399?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_13_layer_call_fn_103409
+__inference_dropout_13_layer_call_fn_103414?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_103425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_103434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103467
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103500?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_5_layer_call_fn_103517
,__inference_lstm_cell_5_layer_call_fn_103534?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101205v
4-./0123A?>
7?4
*?'
input_22?????????F
p

 
? "%?"
?
0?????????F
? ?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101233v
4-./0123A?>
7?4
*?'
input_22?????????F
p 

 
? "%?"
?
0?????????F
? ?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101667t
4-./0123??<
5?2
(?%
inputs?????????F
p

 
? "%?"
?
0?????????F
? ?
D__inference_T-GCN-WX_layer_call_and_return_conditional_losses_101945t
4-./0123??<
5?2
(?%
inputs?????????F
p 

 
? "%?"
?
0?????????F
? ?
)__inference_T-GCN-WX_layer_call_fn_101287i
4-./0123A?>
7?4
*?'
input_22?????????F
p

 
? "??????????F?
)__inference_T-GCN-WX_layer_call_fn_101340i
4-./0123A?>
7?4
*?'
input_22?????????F
p 

 
? "??????????F?
)__inference_T-GCN-WX_layer_call_fn_101970g
4-./0123??<
5?2
(?%
inputs?????????F
p

 
? "??????????F?
)__inference_T-GCN-WX_layer_call_fn_101995g
4-./0123??<
5?2
(?%
inputs?????????F
p 

 
? "??????????F?
 __inference__wrapped_model_99751|
4-./01239?6
/?,
*?'
input_22?????????F
? "3?0
.
model_15"?
model_15?????????F?
D__inference_dense_12_layer_call_and_return_conditional_losses_103425\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????F
? |
)__inference_dense_12_layer_call_fn_103434O23/?,
%?"
 ?
inputs?????????d
? "??????????F?
D__inference_dense_13_layer_call_and_return_conditional_losses_102025l7?4
-?*
(?%
inputs?????????F
? "-?*
#? 
0?????????F
? ?
)__inference_dense_13_layer_call_fn_102034_7?4
-?*
(?%
inputs?????????F
? " ??????????F?
F__inference_dropout_12_layer_call_and_return_conditional_losses_102046l;?8
1?.
(?%
inputs?????????F
p
? "-?*
#? 
0?????????F
? ?
F__inference_dropout_12_layer_call_and_return_conditional_losses_102051l;?8
1?.
(?%
inputs?????????F
p 
? "-?*
#? 
0?????????F
? ?
+__inference_dropout_12_layer_call_fn_102056_;?8
1?.
(?%
inputs?????????F
p
? " ??????????F?
+__inference_dropout_12_layer_call_fn_102061_;?8
1?.
(?%
inputs?????????F
p 
? " ??????????F?
F__inference_dropout_13_layer_call_and_return_conditional_losses_103399\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
F__inference_dropout_13_layer_call_and_return_conditional_losses_103404\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ~
+__inference_dropout_13_layer_call_fn_103409O3?0
)?&
 ?
inputs?????????d
p
? "??????????d~
+__inference_dropout_13_layer_call_fn_103414O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
___inference_fixed_adjacency_graph_convolution_5_layer_call_and_return_conditional_losses_102683g4-.5?2
+?(
&?#
features?????????F
? ")?&
?
0?????????F
? ?
D__inference_fixed_adjacency_graph_convolution_5_layer_call_fn_102694Z4-.5?2
+?(
&?#
features?????????F
? "??????????F?
B__inference_lstm_5_layer_call_and_return_conditional_losses_102884}/01O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103037}/01O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p 

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103212m/01??<
5?2
$?!
inputs?????????F

 
p

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_103365m/01??<
5?2
$?!
inputs?????????F

 
p 

 
? "%?"
?
0?????????d
? ?
'__inference_lstm_5_layer_call_fn_103048p/01O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p

 
? "??????????d?
'__inference_lstm_5_layer_call_fn_103059p/01O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p 

 
? "??????????d?
'__inference_lstm_5_layer_call_fn_103376`/01??<
5?2
$?!
inputs?????????F

 
p

 
? "??????????d?
'__inference_lstm_5_layer_call_fn_103387`/01??<
5?2
$?!
inputs?????????F

 
p 

 
? "??????????d?
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103467?/01??}
v?s
 ?
inputs?????????F
K?H
"?
states/0?????????d
"?
states/1?????????d
p
? "s?p
i?f
?
0/0?????????d
E?B
?
0/1/0?????????d
?
0/1/1?????????d
? ?
G__inference_lstm_cell_5_layer_call_and_return_conditional_losses_103500?/01??}
v?s
 ?
inputs?????????F
K?H
"?
states/0?????????d
"?
states/1?????????d
p 
? "s?p
i?f
?
0/0?????????d
E?B
?
0/1/0?????????d
?
0/1/1?????????d
? ?
,__inference_lstm_cell_5_layer_call_fn_103517?/01??}
v?s
 ?
inputs?????????F
K?H
"?
states/0?????????d
"?
states/1?????????d
p
? "c?`
?
0?????????d
A?>
?
1/0?????????d
?
1/1?????????d?
,__inference_lstm_cell_5_layer_call_fn_103534?/01??}
v?s
 ?
inputs?????????F
K?H
"?
states/0?????????d
"?
states/1?????????d
p 
? "c?`
?
0?????????d
A?>
?
1/0?????????d
?
1/1?????????d?
D__inference_model_15_layer_call_and_return_conditional_losses_100911p4-./0123=?:
3?0
&?#
input_23?????????F
p

 
? "%?"
?
0?????????F
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_100941p4-./0123=?:
3?0
&?#
input_23?????????F
p 

 
? "%?"
?
0?????????F
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_102328n4-./0123;?8
1?.
$?!
inputs?????????F
p

 
? "%?"
?
0?????????F
? ?
D__inference_model_15_layer_call_and_return_conditional_losses_102570n4-./0123;?8
1?.
$?!
inputs?????????F
p 

 
? "%?"
?
0?????????F
? ?
)__inference_model_15_layer_call_fn_100993c4-./0123=?:
3?0
&?#
input_23?????????F
p

 
? "??????????F?
)__inference_model_15_layer_call_fn_101044c4-./0123=?:
3?0
&?#
input_23?????????F
p 

 
? "??????????F?
)__inference_model_15_layer_call_fn_102591a4-./0123;?8
1?.
$?!
inputs?????????F
p

 
? "??????????F?
)__inference_model_15_layer_call_fn_102612a4-./0123;?8
1?.
$?!
inputs?????????F
p 

 
? "??????????F?
D__inference_permute_7_layer_call_and_return_conditional_losses_99758?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_permute_7_layer_call_fn_99764?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_reshape_28_layer_call_and_return_conditional_losses_102074d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
+__inference_reshape_28_layer_call_fn_102079W7?4
-?*
(?%
inputs?????????F
? "??????????F?
F__inference_reshape_29_layer_call_and_return_conditional_losses_102625d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
+__inference_reshape_29_layer_call_fn_102630W7?4
-?*
(?%
inputs?????????F
? "??????????F?
F__inference_reshape_30_layer_call_and_return_conditional_losses_102708d3?0
)?&
$?!
inputs?????????F
? "-?*
#? 
0?????????F
? ?
+__inference_reshape_30_layer_call_fn_102713W3?0
)?&
$?!
inputs?????????F
? " ??????????F?
F__inference_reshape_31_layer_call_and_return_conditional_losses_102726d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
+__inference_reshape_31_layer_call_fn_102731W7?4
-?*
(?%
inputs?????????F
? "??????????F?
$__inference_signature_wrapper_101375?
4-./0123E?B
? 
;?8
6
input_22*?'
input_22?????????F"3?0
.
model_15"?
model_15?????????F