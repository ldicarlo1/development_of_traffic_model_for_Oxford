??
??
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
%fixed_adjacency_graph_convolution_8/AVarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*6
shared_name'%fixed_adjacency_graph_convolution_8/A
?
9fixed_adjacency_graph_convolution_8/A/Read/ReadVariableOpReadVariableOp%fixed_adjacency_graph_convolution_8/A*
_output_shapes

:FF*
dtype0
?
*fixed_adjacency_graph_convolution_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*fixed_adjacency_graph_convolution_8/kernel
?
>fixed_adjacency_graph_convolution_8/kernel/Read/ReadVariableOpReadVariableOp*fixed_adjacency_graph_convolution_8/kernel*
_output_shapes

:
*
dtype0
?
(fixed_adjacency_graph_convolution_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*9
shared_name*(fixed_adjacency_graph_convolution_8/bias
?
<fixed_adjacency_graph_convolution_8/bias/Read/ReadVariableOpReadVariableOp(fixed_adjacency_graph_convolution_8/bias*
_output_shapes

:F*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?F*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
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
?
lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?**
shared_namelstm_8/lstm_cell_8/kernel
?
-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/kernel*
_output_shapes
:	F?*
dtype0
?
#lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_8/lstm_cell_8/recurrent_kernel
?
7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_8/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_8/lstm_cell_8/bias
?
+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/bias*
_output_shapes	
:?*
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
1Adam/fixed_adjacency_graph_convolution_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31Adam/fixed_adjacency_graph_convolution_8/kernel/m
?
EAdam/fixed_adjacency_graph_convolution_8/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_8/kernel/m*
_output_shapes

:
*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_8/bias/m
?
CAdam/fixed_adjacency_graph_convolution_8/bias/m/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_8/bias/m*
_output_shapes

:F*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?F*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:F*
dtype0
?
 Adam/lstm_8/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/m
?
4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/m*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
?
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_8/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_8/lstm_cell_8/bias/m
?
2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/m*
_output_shapes	
:?*
dtype0
?
1Adam/fixed_adjacency_graph_convolution_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31Adam/fixed_adjacency_graph_convolution_8/kernel/v
?
EAdam/fixed_adjacency_graph_convolution_8/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_8/kernel/v*
_output_shapes

:
*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_8/bias/v
?
CAdam/fixed_adjacency_graph_convolution_8/bias/v/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_8/bias/v*
_output_shapes

:F*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?F*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:F*
dtype0
?
 Adam/lstm_8/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/v
?
4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/v*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
?
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_8/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_8/lstm_cell_8/bias/v
?
2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?9B?9 B?9
?
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
?
9iter

:beta_1

;beta_2
	<decay
=learning_ratem?m?3m?4m?>m??m?@m?v?v?3v?4v?>v??v?@v?
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
?
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
?
regularization_losses

Flayers
trainable_variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
	variables
lj
VARIABLE_VALUE%fixed_adjacency_graph_convolution_8/A1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE*fixed_adjacency_graph_convolution_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE(fixed_adjacency_graph_convolution_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
?
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
?
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
?
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
?
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
?
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
?
/regularization_losses

ilayers
0trainable_variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
1	variables
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
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
_]
VARIABLE_VALUElstm_8/lstm_cell_8/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_8/lstm_cell_8/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_8/lstm_cell_8/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
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
?
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
?
_fn_kwargs
?	variables
?	keras_api
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
?	variables
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_17Placeholder*+
_output_shapes
:?????????F*
dtype0* 
shape:?????????F
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17%fixed_adjacency_graph_convolution_8/A*fixed_adjacency_graph_convolution_8/kernel(fixed_adjacency_graph_convolution_8/biaslstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biasdense_8/kerneldense_8/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_166772
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9fixed_adjacency_graph_convolution_8/A/Read/ReadVariableOp>fixed_adjacency_graph_convolution_8/kernel/Read/ReadVariableOp<fixed_adjacency_graph_convolution_8/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_8/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_8/kernel/m/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_8/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_8/kernel/v/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_8/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpConst*,
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
GPU 2J 8? *(
f#R!
__inference__traced_save_168315
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%fixed_adjacency_graph_convolution_8/A*fixed_adjacency_graph_convolution_8/kernel(fixed_adjacency_graph_convolution_8/biasdense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biastotalcounttotal_1count_11Adam/fixed_adjacency_graph_convolution_8/kernel/m/Adam/fixed_adjacency_graph_convolution_8/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/m Adam/lstm_8/lstm_cell_8/kernel/m*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mAdam/lstm_8/lstm_cell_8/bias/m1Adam/fixed_adjacency_graph_convolution_8/kernel/v/Adam/fixed_adjacency_graph_convolution_8/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/v Adam/lstm_8/lstm_cell_8/kernel/v*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vAdam/lstm_8/lstm_cell_8/bias/v*+
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_168418??
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_167017

inputsG
Cfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_8_add_readvariableop_resource5
1lstm_8_lstm_cell_8_matmul_readvariableop_resource7
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource6
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_8/add/ReadVariableOp?>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?(lstm_8/lstm_cell_8/MatMul/ReadVariableOp?*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?lstm_8/while?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinputs(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDimsx
reshape_24/ShapeShape$tf.expand_dims_8/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_24/Shape?
reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_24/strided_slice/stack?
 reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_1?
 reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_2?
reshape_24/strided_sliceStridedSlicereshape_24/Shape:output:0'reshape_24/strided_slice/stack:output:0)reshape_24/strided_slice/stack_1:output:0)reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_24/strided_slicez
reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_24/Reshape/shape/1z
reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/2?
reshape_24/Reshape/shapePack!reshape_24/strided_slice:output:0#reshape_24/Reshape/shape/1:output:0#reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_24/Reshape/shape?
reshape_24/ReshapeReshape$tf.expand_dims_8/ExpandDims:output:0!reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_24/Reshape?
2fixed_adjacency_graph_convolution_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_8/transpose/perm?
-fixed_adjacency_graph_convolution_8/transpose	Transposereshape_24/Reshape:output:0;fixed_adjacency_graph_convolution_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_8/transpose?
)fixed_adjacency_graph_convolution_8/ShapeShape1fixed_adjacency_graph_convolution_8/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_8/Shape?
+fixed_adjacency_graph_convolution_8/unstackUnpack2fixed_adjacency_graph_convolution_8/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_8/unstack?
:fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_8/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_8/Shape_1?
-fixed_adjacency_graph_convolution_8/unstack_1Unpack4fixed_adjacency_graph_convolution_8/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_1?
1fixed_adjacency_graph_convolution_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_8/Reshape/shape?
+fixed_adjacency_graph_convolution_8/ReshapeReshape1fixed_adjacency_graph_convolution_8/transpose:y:0:fixed_adjacency_graph_convolution_8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_8/Reshape?
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_8/transpose_1/perm?
/fixed_adjacency_graph_convolution_8/transpose_1	TransposeFfixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_8/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_8/transpose_1?
3fixed_adjacency_graph_convolution_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_8/Reshape_1/shape?
-fixed_adjacency_graph_convolution_8/Reshape_1Reshape3fixed_adjacency_graph_convolution_8/transpose_1:y:0<fixed_adjacency_graph_convolution_8/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_8/Reshape_1?
*fixed_adjacency_graph_convolution_8/MatMulMatMul4fixed_adjacency_graph_convolution_8/Reshape:output:06fixed_adjacency_graph_convolution_8/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_8/MatMul?
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_8/Reshape_2/shapePack4fixed_adjacency_graph_convolution_8/unstack:output:0>fixed_adjacency_graph_convolution_8/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_8/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_8/Reshape_2/shape?
-fixed_adjacency_graph_convolution_8/Reshape_2Reshape4fixed_adjacency_graph_convolution_8/MatMul:product:0<fixed_adjacency_graph_convolution_8/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_8/Reshape_2?
4fixed_adjacency_graph_convolution_8/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_8/transpose_2/perm?
/fixed_adjacency_graph_convolution_8/transpose_2	Transpose6fixed_adjacency_graph_convolution_8/Reshape_2:output:0=fixed_adjacency_graph_convolution_8/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_8/transpose_2?
+fixed_adjacency_graph_convolution_8/Shape_2Shape3fixed_adjacency_graph_convolution_8/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_8/Shape_2?
-fixed_adjacency_graph_convolution_8/unstack_2Unpack4fixed_adjacency_graph_convolution_8/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_2?
:fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02<
:fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_8/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2-
+fixed_adjacency_graph_convolution_8/Shape_3?
-fixed_adjacency_graph_convolution_8/unstack_3Unpack4fixed_adjacency_graph_convolution_8/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_3?
3fixed_adjacency_graph_convolution_8/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_8/Reshape_3/shape?
-fixed_adjacency_graph_convolution_8/Reshape_3Reshape3fixed_adjacency_graph_convolution_8/transpose_2:y:0<fixed_adjacency_graph_convolution_8/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_8/Reshape_3?
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02@
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_8/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_8/transpose_3/perm?
/fixed_adjacency_graph_convolution_8/transpose_3	TransposeFfixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_8/transpose_3/perm:output:0*
T0*
_output_shapes

:
21
/fixed_adjacency_graph_convolution_8/transpose_3?
3fixed_adjacency_graph_convolution_8/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_8/Reshape_4/shape?
-fixed_adjacency_graph_convolution_8/Reshape_4Reshape3fixed_adjacency_graph_convolution_8/transpose_3:y:0<fixed_adjacency_graph_convolution_8/Reshape_4/shape:output:0*
T0*
_output_shapes

:
2/
-fixed_adjacency_graph_convolution_8/Reshape_4?
,fixed_adjacency_graph_convolution_8/MatMul_1MatMul6fixed_adjacency_graph_convolution_8/Reshape_3:output:06fixed_adjacency_graph_convolution_8/Reshape_4:output:0*
T0*'
_output_shapes
:?????????
2.
,fixed_adjacency_graph_convolution_8/MatMul_1?
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
27
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_8/Reshape_5/shapePack6fixed_adjacency_graph_convolution_8/unstack_2:output:0>fixed_adjacency_graph_convolution_8/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_8/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_8/Reshape_5/shape?
-fixed_adjacency_graph_convolution_8/Reshape_5Reshape6fixed_adjacency_graph_convolution_8/MatMul_1:product:0<fixed_adjacency_graph_convolution_8/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F
2/
-fixed_adjacency_graph_convolution_8/Reshape_5?
6fixed_adjacency_graph_convolution_8/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_8_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_8/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_8/addAddV26fixed_adjacency_graph_convolution_8/Reshape_5:output:0>fixed_adjacency_graph_convolution_8/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F
2)
'fixed_adjacency_graph_convolution_8/add
reshape_25/ShapeShape+fixed_adjacency_graph_convolution_8/add:z:0*
T0*
_output_shapes
:2
reshape_25/Shape?
reshape_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_25/strided_slice/stack?
 reshape_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_25/strided_slice/stack_1?
 reshape_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_25/strided_slice/stack_2?
reshape_25/strided_sliceStridedSlicereshape_25/Shape:output:0'reshape_25/strided_slice/stack:output:0)reshape_25/strided_slice/stack_1:output:0)reshape_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_25/strided_slicez
reshape_25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_25/Reshape/shape/1?
reshape_25/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_25/Reshape/shape/2z
reshape_25/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_25/Reshape/shape/3?
reshape_25/Reshape/shapePack!reshape_25/strided_slice:output:0#reshape_25/Reshape/shape/1:output:0#reshape_25/Reshape/shape/2:output:0#reshape_25/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_25/Reshape/shape?
reshape_25/ReshapeReshape+fixed_adjacency_graph_convolution_8/add:z:0!reshape_25/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F
2
reshape_25/Reshape?
permute_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_8/transpose/perm?
permute_8/transpose	Transposereshape_25/Reshape:output:0!permute_8/transpose/perm:output:0*
T0*/
_output_shapes
:?????????
F2
permute_8/transposek
reshape_26/ShapeShapepermute_8/transpose:y:0*
T0*
_output_shapes
:2
reshape_26/Shape?
reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_26/strided_slice/stack?
 reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_1?
 reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_2?
reshape_26/strided_sliceStridedSlicereshape_26/Shape:output:0'reshape_26/strided_slice/stack:output:0)reshape_26/strided_slice/stack_1:output:0)reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_26/strided_slice?
reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_26/Reshape/shape/1z
reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_26/Reshape/shape/2?
reshape_26/Reshape/shapePack!reshape_26/strided_slice:output:0#reshape_26/Reshape/shape/1:output:0#reshape_26/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_26/Reshape/shape?
reshape_26/ReshapeReshapepermute_8/transpose:y:0!reshape_26/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
F2
reshape_26/Reshapeg
lstm_8/ShapeShapereshape_26/Reshape:output:0*
T0*
_output_shapes
:2
lstm_8/Shape?
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack?
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1?
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2?
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicek
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/mul/y?
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/Less/y?
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessq
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/packed/1?
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const?
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/zeroso
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/mul/y?
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/Less/y?
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lessu
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/packed/1?
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const?
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/zeros_1?
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm?
lstm_8/transpose	Transposereshape_26/Reshape:output:0lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????F2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1?
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack?
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1?
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2?
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1?
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_8/TensorArrayV2/element_shape?
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2?
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor?
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack?
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1?
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2?
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_8/strided_slice_2?
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp?
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/MatMul?
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/MatMul_1?
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/add?
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/BiasAddv
lstm_8/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/lstm_cell_8/Const?
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dim?
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_8/lstm_cell_8/split?
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid?
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid_1?
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul?
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0!lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul_1?
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/add_1?
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid_2?
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0lstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul_2?
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$lstm_8/TensorArrayV2_1/element_shape?
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time?
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counter?
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_8_while_body_166919*$
condR
lstm_8_while_cond_166918*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_8/while?
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack?
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_8/strided_slice_3/stack?
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1?
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2?
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_8/strided_slice_3?
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm?
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimew
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMullstm_8/strided_slice_3:output:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShapelstm_8/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_8/dropout/Mul_1?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_8/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_8/BiasAddy
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_8/Sigmoid?
IdentityIdentitydense_8/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_8/add/ReadVariableOp?^fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_8/add/ReadVariableOp6fixed_adjacency_graph_convolution_8/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?@
?
while_body_167482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
 model_8_lstm_8_while_body_165369:
6model_8_lstm_8_while_model_8_lstm_8_while_loop_counter@
<model_8_lstm_8_while_model_8_lstm_8_while_maximum_iterations$
 model_8_lstm_8_while_placeholder&
"model_8_lstm_8_while_placeholder_1&
"model_8_lstm_8_while_placeholder_2&
"model_8_lstm_8_while_placeholder_39
5model_8_lstm_8_while_model_8_lstm_8_strided_slice_1_0u
qmodel_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0E
Amodel_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0G
Cmodel_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0F
Bmodel_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0!
model_8_lstm_8_while_identity#
model_8_lstm_8_while_identity_1#
model_8_lstm_8_while_identity_2#
model_8_lstm_8_while_identity_3#
model_8_lstm_8_while_identity_4#
model_8_lstm_8_while_identity_57
3model_8_lstm_8_while_model_8_lstm_8_strided_slice_1s
omodel_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_8_tensorarrayunstack_tensorlistfromtensorC
?model_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resourceE
Amodel_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resourceD
@model_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource??7model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?6model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?8model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
Fmodel_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2H
Fmodel_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8model_8/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0 model_8_lstm_8_while_placeholderOmodel_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02:
8model_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem?
6model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpAmodel_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype028
6model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?
'model_8/lstm_8/while/lstm_cell_8/MatMulMatMul?model_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0>model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'model_8/lstm_8/while/lstm_cell_8/MatMul?
8model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpCmodel_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02:
8model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
)model_8/lstm_8/while/lstm_cell_8/MatMul_1MatMul"model_8_lstm_8_while_placeholder_2@model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_8/lstm_8/while/lstm_cell_8/MatMul_1?
$model_8/lstm_8/while/lstm_cell_8/addAddV21model_8/lstm_8/while/lstm_cell_8/MatMul:product:03model_8/lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$model_8/lstm_8/while/lstm_cell_8/add?
7model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpBmodel_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype029
7model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?
(model_8/lstm_8/while/lstm_cell_8/BiasAddBiasAdd(model_8/lstm_8/while/lstm_cell_8/add:z:0?model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_8/lstm_8/while/lstm_cell_8/BiasAdd?
&model_8/lstm_8/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_8/lstm_8/while/lstm_cell_8/Const?
0model_8/lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_8/lstm_8/while/lstm_cell_8/split/split_dim?
&model_8/lstm_8/while/lstm_cell_8/splitSplit9model_8/lstm_8/while/lstm_cell_8/split/split_dim:output:01model_8/lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2(
&model_8/lstm_8/while/lstm_cell_8/split?
(model_8/lstm_8/while/lstm_cell_8/SigmoidSigmoid/model_8/lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2*
(model_8/lstm_8/while/lstm_cell_8/Sigmoid?
*model_8/lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid/model_8/lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2,
*model_8/lstm_8/while/lstm_cell_8/Sigmoid_1?
$model_8/lstm_8/while/lstm_cell_8/mulMul.model_8/lstm_8/while/lstm_cell_8/Sigmoid_1:y:0"model_8_lstm_8_while_placeholder_3*
T0*(
_output_shapes
:??????????2&
$model_8/lstm_8/while/lstm_cell_8/mul?
&model_8/lstm_8/while/lstm_cell_8/mul_1Mul,model_8/lstm_8/while/lstm_cell_8/Sigmoid:y:0/model_8/lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2(
&model_8/lstm_8/while/lstm_cell_8/mul_1?
&model_8/lstm_8/while/lstm_cell_8/add_1AddV2(model_8/lstm_8/while/lstm_cell_8/mul:z:0*model_8/lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_8/lstm_8/while/lstm_cell_8/add_1?
*model_8/lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid/model_8/lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2,
*model_8/lstm_8/while/lstm_cell_8/Sigmoid_2?
&model_8/lstm_8/while/lstm_cell_8/mul_2Mul.model_8/lstm_8/while/lstm_cell_8/Sigmoid_2:y:0*model_8/lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_8/lstm_8/while/lstm_cell_8/mul_2?
9model_8/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_8_lstm_8_while_placeholder_1 model_8_lstm_8_while_placeholder*model_8/lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9model_8/lstm_8/while/TensorArrayV2Write/TensorListSetItemz
model_8/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_8/while/add/y?
model_8/lstm_8/while/addAddV2 model_8_lstm_8_while_placeholder#model_8/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/while/add~
model_8/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_8/lstm_8/while/add_1/y?
model_8/lstm_8/while/add_1AddV26model_8_lstm_8_while_model_8_lstm_8_while_loop_counter%model_8/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/while/add_1?
model_8/lstm_8/while/IdentityIdentitymodel_8/lstm_8/while/add_1:z:08^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_8/lstm_8/while/Identity?
model_8/lstm_8/while/Identity_1Identity<model_8_lstm_8_while_model_8_lstm_8_while_maximum_iterations8^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_8/while/Identity_1?
model_8/lstm_8/while/Identity_2Identitymodel_8/lstm_8/while/add:z:08^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_8/while/Identity_2?
model_8/lstm_8/while/Identity_3IdentityImodel_8/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_8/lstm_8/while/Identity_3?
model_8/lstm_8/while/Identity_4Identity*model_8/lstm_8/while/lstm_cell_8/mul_2:z:08^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_8/lstm_8/while/Identity_4?
model_8/lstm_8/while/Identity_5Identity*model_8/lstm_8/while/lstm_cell_8/add_1:z:08^model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7^model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp9^model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_8/lstm_8/while/Identity_5"G
model_8_lstm_8_while_identity&model_8/lstm_8/while/Identity:output:0"K
model_8_lstm_8_while_identity_1(model_8/lstm_8/while/Identity_1:output:0"K
model_8_lstm_8_while_identity_2(model_8/lstm_8/while/Identity_2:output:0"K
model_8_lstm_8_while_identity_3(model_8/lstm_8/while/Identity_3:output:0"K
model_8_lstm_8_while_identity_4(model_8/lstm_8/while/Identity_4:output:0"K
model_8_lstm_8_while_identity_5(model_8/lstm_8/while/Identity_5:output:0"?
@model_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resourceBmodel_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"?
Amodel_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resourceCmodel_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"?
?model_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resourceAmodel_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"l
3model_8_lstm_8_while_model_8_lstm_8_strided_slice_15model_8_lstm_8_while_model_8_lstm_8_strided_slice_1_0"?
omodel_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_8_tensorarrayunstack_tensorlistfromtensorqmodel_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_model_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2r
7model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp7model_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2p
6model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp6model_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2t
8model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp8model_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_168068

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
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
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_lstm_cell_8_layer_call_fn_168199

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
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_166562

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
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
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_lstm_8_layer_call_fn_167725
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
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1659382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?
?
'__inference_lstm_8_layer_call_fn_168056

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1665202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
c
*__inference_dropout_8_layer_call_fn_168078

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_166287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_166287___redundant_placeholder04
0while_while_cond_166287___redundant_placeholder14
0while_while_cond_166287___redundant_placeholder24
0while_while_cond_166287___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__wrapped_model_165460
input_17O
Kmodel_8_fixed_adjacency_graph_convolution_8_shape_1_readvariableop_resourceO
Kmodel_8_fixed_adjacency_graph_convolution_8_shape_3_readvariableop_resourceK
Gmodel_8_fixed_adjacency_graph_convolution_8_add_readvariableop_resource=
9model_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource?
;model_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource>
:model_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource2
.model_8_dense_8_matmul_readvariableop_resource3
/model_8_dense_8_biasadd_readvariableop_resource
identity??&model_8/dense_8/BiasAdd/ReadVariableOp?%model_8/dense_8/MatMul/ReadVariableOp?>model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOp?Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?1model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?0model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp?2model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?model_8/lstm_8/while?
'model_8/tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_8/tf.expand_dims_8/ExpandDims/dim?
#model_8/tf.expand_dims_8/ExpandDims
ExpandDimsinput_170model_8/tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2%
#model_8/tf.expand_dims_8/ExpandDims?
model_8/reshape_24/ShapeShape,model_8/tf.expand_dims_8/ExpandDims:output:0*
T0*
_output_shapes
:2
model_8/reshape_24/Shape?
&model_8/reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_24/strided_slice/stack?
(model_8/reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_24/strided_slice/stack_1?
(model_8/reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_24/strided_slice/stack_2?
 model_8/reshape_24/strided_sliceStridedSlice!model_8/reshape_24/Shape:output:0/model_8/reshape_24/strided_slice/stack:output:01model_8/reshape_24/strided_slice/stack_1:output:01model_8/reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_24/strided_slice?
"model_8/reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_24/Reshape/shape/1?
"model_8/reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_24/Reshape/shape/2?
 model_8/reshape_24/Reshape/shapePack)model_8/reshape_24/strided_slice:output:0+model_8/reshape_24/Reshape/shape/1:output:0+model_8/reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_24/Reshape/shape?
model_8/reshape_24/ReshapeReshape,model_8/tf.expand_dims_8/ExpandDims:output:0)model_8/reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_8/reshape_24/Reshape?
:model_8/fixed_adjacency_graph_convolution_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:model_8/fixed_adjacency_graph_convolution_8/transpose/perm?
5model_8/fixed_adjacency_graph_convolution_8/transpose	Transpose#model_8/reshape_24/Reshape:output:0Cmodel_8/fixed_adjacency_graph_convolution_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F27
5model_8/fixed_adjacency_graph_convolution_8/transpose?
1model_8/fixed_adjacency_graph_convolution_8/ShapeShape9model_8/fixed_adjacency_graph_convolution_8/transpose:y:0*
T0*
_output_shapes
:23
1model_8/fixed_adjacency_graph_convolution_8/Shape?
3model_8/fixed_adjacency_graph_convolution_8/unstackUnpack:model_8/fixed_adjacency_graph_convolution_8/Shape:output:0*
T0*
_output_shapes
: : : *	
num25
3model_8/fixed_adjacency_graph_convolution_8/unstack?
Bmodel_8/fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOp?
3model_8/fixed_adjacency_graph_convolution_8/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   25
3model_8/fixed_adjacency_graph_convolution_8/Shape_1?
5model_8/fixed_adjacency_graph_convolution_8/unstack_1Unpack<model_8/fixed_adjacency_graph_convolution_8/Shape_1:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_8/unstack_1?
9model_8/fixed_adjacency_graph_convolution_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2;
9model_8/fixed_adjacency_graph_convolution_8/Reshape/shape?
3model_8/fixed_adjacency_graph_convolution_8/ReshapeReshape9model_8/fixed_adjacency_graph_convolution_8/transpose:y:0Bmodel_8/fixed_adjacency_graph_convolution_8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F25
3model_8/fixed_adjacency_graph_convolution_8/Reshape?
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?
<model_8/fixed_adjacency_graph_convolution_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_8/transpose_1/perm?
7model_8/fixed_adjacency_graph_convolution_8/transpose_1	TransposeNmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_8/transpose_1/perm:output:0*
T0*
_output_shapes

:FF29
7model_8/fixed_adjacency_graph_convolution_8/transpose_1?
;model_8/fixed_adjacency_graph_convolution_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2=
;model_8/fixed_adjacency_graph_convolution_8/Reshape_1/shape?
5model_8/fixed_adjacency_graph_convolution_8/Reshape_1Reshape;model_8/fixed_adjacency_graph_convolution_8/transpose_1:y:0Dmodel_8/fixed_adjacency_graph_convolution_8/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF27
5model_8/fixed_adjacency_graph_convolution_8/Reshape_1?
2model_8/fixed_adjacency_graph_convolution_8/MatMulMatMul<model_8/fixed_adjacency_graph_convolution_8/Reshape:output:0>model_8/fixed_adjacency_graph_convolution_8/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F24
2model_8/fixed_adjacency_graph_convolution_8/MatMul?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/1?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/2?
;model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shapePack<model_8/fixed_adjacency_graph_convolution_8/unstack:output:0Fmodel_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape?
5model_8/fixed_adjacency_graph_convolution_8/Reshape_2Reshape<model_8/fixed_adjacency_graph_convolution_8/MatMul:product:0Dmodel_8/fixed_adjacency_graph_convolution_8/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F27
5model_8/fixed_adjacency_graph_convolution_8/Reshape_2?
<model_8/fixed_adjacency_graph_convolution_8/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_8/fixed_adjacency_graph_convolution_8/transpose_2/perm?
7model_8/fixed_adjacency_graph_convolution_8/transpose_2	Transpose>model_8/fixed_adjacency_graph_convolution_8/Reshape_2:output:0Emodel_8/fixed_adjacency_graph_convolution_8/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F29
7model_8/fixed_adjacency_graph_convolution_8/transpose_2?
3model_8/fixed_adjacency_graph_convolution_8/Shape_2Shape;model_8/fixed_adjacency_graph_convolution_8/transpose_2:y:0*
T0*
_output_shapes
:25
3model_8/fixed_adjacency_graph_convolution_8/Shape_2?
5model_8/fixed_adjacency_graph_convolution_8/unstack_2Unpack<model_8/fixed_adjacency_graph_convolution_8/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num27
5model_8/fixed_adjacency_graph_convolution_8/unstack_2?
Bmodel_8/fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02D
Bmodel_8/fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOp?
3model_8/fixed_adjacency_graph_convolution_8/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   25
3model_8/fixed_adjacency_graph_convolution_8/Shape_3?
5model_8/fixed_adjacency_graph_convolution_8/unstack_3Unpack<model_8/fixed_adjacency_graph_convolution_8/Shape_3:output:0*
T0*
_output_shapes
: : *	
num27
5model_8/fixed_adjacency_graph_convolution_8/unstack_3?
;model_8/fixed_adjacency_graph_convolution_8/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;model_8/fixed_adjacency_graph_convolution_8/Reshape_3/shape?
5model_8/fixed_adjacency_graph_convolution_8/Reshape_3Reshape;model_8/fixed_adjacency_graph_convolution_8/transpose_2:y:0Dmodel_8/fixed_adjacency_graph_convolution_8/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????27
5model_8/fixed_adjacency_graph_convolution_8/Reshape_3?
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOpReadVariableOpKmodel_8_fixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02H
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?
<model_8/fixed_adjacency_graph_convolution_8/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_8/fixed_adjacency_graph_convolution_8/transpose_3/perm?
7model_8/fixed_adjacency_graph_convolution_8/transpose_3	TransposeNmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp:value:0Emodel_8/fixed_adjacency_graph_convolution_8/transpose_3/perm:output:0*
T0*
_output_shapes

:
29
7model_8/fixed_adjacency_graph_convolution_8/transpose_3?
;model_8/fixed_adjacency_graph_convolution_8/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2=
;model_8/fixed_adjacency_graph_convolution_8/Reshape_4/shape?
5model_8/fixed_adjacency_graph_convolution_8/Reshape_4Reshape;model_8/fixed_adjacency_graph_convolution_8/transpose_3:y:0Dmodel_8/fixed_adjacency_graph_convolution_8/Reshape_4/shape:output:0*
T0*
_output_shapes

:
27
5model_8/fixed_adjacency_graph_convolution_8/Reshape_4?
4model_8/fixed_adjacency_graph_convolution_8/MatMul_1MatMul>model_8/fixed_adjacency_graph_convolution_8/Reshape_3:output:0>model_8/fixed_adjacency_graph_convolution_8/Reshape_4:output:0*
T0*'
_output_shapes
:?????????
26
4model_8/fixed_adjacency_graph_convolution_8/MatMul_1?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/1?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2?
=model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/2?
;model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shapePack>model_8/fixed_adjacency_graph_convolution_8/unstack_2:output:0Fmodel_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/1:output:0Fmodel_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape?
5model_8/fixed_adjacency_graph_convolution_8/Reshape_5Reshape>model_8/fixed_adjacency_graph_convolution_8/MatMul_1:product:0Dmodel_8/fixed_adjacency_graph_convolution_8/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F
27
5model_8/fixed_adjacency_graph_convolution_8/Reshape_5?
>model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOpReadVariableOpGmodel_8_fixed_adjacency_graph_convolution_8_add_readvariableop_resource*
_output_shapes

:F*
dtype02@
>model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOp?
/model_8/fixed_adjacency_graph_convolution_8/addAddV2>model_8/fixed_adjacency_graph_convolution_8/Reshape_5:output:0Fmodel_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F
21
/model_8/fixed_adjacency_graph_convolution_8/add?
model_8/reshape_25/ShapeShape3model_8/fixed_adjacency_graph_convolution_8/add:z:0*
T0*
_output_shapes
:2
model_8/reshape_25/Shape?
&model_8/reshape_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_25/strided_slice/stack?
(model_8/reshape_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_25/strided_slice/stack_1?
(model_8/reshape_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_25/strided_slice/stack_2?
 model_8/reshape_25/strided_sliceStridedSlice!model_8/reshape_25/Shape:output:0/model_8/reshape_25/strided_slice/stack:output:01model_8/reshape_25/strided_slice/stack_1:output:01model_8/reshape_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_25/strided_slice?
"model_8/reshape_25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_25/Reshape/shape/1?
"model_8/reshape_25/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_8/reshape_25/Reshape/shape/2?
"model_8/reshape_25/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_8/reshape_25/Reshape/shape/3?
 model_8/reshape_25/Reshape/shapePack)model_8/reshape_25/strided_slice:output:0+model_8/reshape_25/Reshape/shape/1:output:0+model_8/reshape_25/Reshape/shape/2:output:0+model_8/reshape_25/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_25/Reshape/shape?
model_8/reshape_25/ReshapeReshape3model_8/fixed_adjacency_graph_convolution_8/add:z:0)model_8/reshape_25/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F
2
model_8/reshape_25/Reshape?
 model_8/permute_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 model_8/permute_8/transpose/perm?
model_8/permute_8/transpose	Transpose#model_8/reshape_25/Reshape:output:0)model_8/permute_8/transpose/perm:output:0*
T0*/
_output_shapes
:?????????
F2
model_8/permute_8/transpose?
model_8/reshape_26/ShapeShapemodel_8/permute_8/transpose:y:0*
T0*
_output_shapes
:2
model_8/reshape_26/Shape?
&model_8/reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/reshape_26/strided_slice/stack?
(model_8/reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_26/strided_slice/stack_1?
(model_8/reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_8/reshape_26/strided_slice/stack_2?
 model_8/reshape_26/strided_sliceStridedSlice!model_8/reshape_26/Shape:output:0/model_8/reshape_26/strided_slice/stack:output:01model_8/reshape_26/strided_slice/stack_1:output:01model_8/reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_8/reshape_26/strided_slice?
"model_8/reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_8/reshape_26/Reshape/shape/1?
"model_8/reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_8/reshape_26/Reshape/shape/2?
 model_8/reshape_26/Reshape/shapePack)model_8/reshape_26/strided_slice:output:0+model_8/reshape_26/Reshape/shape/1:output:0+model_8/reshape_26/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_8/reshape_26/Reshape/shape?
model_8/reshape_26/ReshapeReshapemodel_8/permute_8/transpose:y:0)model_8/reshape_26/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
F2
model_8/reshape_26/Reshape
model_8/lstm_8/ShapeShape#model_8/reshape_26/Reshape:output:0*
T0*
_output_shapes
:2
model_8/lstm_8/Shape?
"model_8/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_8/lstm_8/strided_slice/stack?
$model_8/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_8/strided_slice/stack_1?
$model_8/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_8/lstm_8/strided_slice/stack_2?
model_8/lstm_8/strided_sliceStridedSlicemodel_8/lstm_8/Shape:output:0+model_8/lstm_8/strided_slice/stack:output:0-model_8/lstm_8/strided_slice/stack_1:output:0-model_8/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_8/lstm_8/strided_slice{
model_8/lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_8/lstm_8/zeros/mul/y?
model_8/lstm_8/zeros/mulMul%model_8/lstm_8/strided_slice:output:0#model_8/lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/zeros/mul}
model_8/lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_8/lstm_8/zeros/Less/y?
model_8/lstm_8/zeros/LessLessmodel_8/lstm_8/zeros/mul:z:0$model_8/lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/zeros/Less?
model_8/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_8/lstm_8/zeros/packed/1?
model_8/lstm_8/zeros/packedPack%model_8/lstm_8/strided_slice:output:0&model_8/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_8/zeros/packed}
model_8/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_8/zeros/Const?
model_8/lstm_8/zerosFill$model_8/lstm_8/zeros/packed:output:0#model_8/lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_8/lstm_8/zeros
model_8/lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_8/lstm_8/zeros_1/mul/y?
model_8/lstm_8/zeros_1/mulMul%model_8/lstm_8/strided_slice:output:0%model_8/lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/zeros_1/mul?
model_8/lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_8/lstm_8/zeros_1/Less/y?
model_8/lstm_8/zeros_1/LessLessmodel_8/lstm_8/zeros_1/mul:z:0&model_8/lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_8/lstm_8/zeros_1/Less?
model_8/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
model_8/lstm_8/zeros_1/packed/1?
model_8/lstm_8/zeros_1/packedPack%model_8/lstm_8/strided_slice:output:0(model_8/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_8/lstm_8/zeros_1/packed?
model_8/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_8/zeros_1/Const?
model_8/lstm_8/zeros_1Fill&model_8/lstm_8/zeros_1/packed:output:0%model_8/lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_8/lstm_8/zeros_1?
model_8/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_8/lstm_8/transpose/perm?
model_8/lstm_8/transpose	Transpose#model_8/reshape_26/Reshape:output:0&model_8/lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????F2
model_8/lstm_8/transpose|
model_8/lstm_8/Shape_1Shapemodel_8/lstm_8/transpose:y:0*
T0*
_output_shapes
:2
model_8/lstm_8/Shape_1?
$model_8/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_8/strided_slice_1/stack?
&model_8/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_8/strided_slice_1/stack_1?
&model_8/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_8/strided_slice_1/stack_2?
model_8/lstm_8/strided_slice_1StridedSlicemodel_8/lstm_8/Shape_1:output:0-model_8/lstm_8/strided_slice_1/stack:output:0/model_8/lstm_8/strided_slice_1/stack_1:output:0/model_8/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_8/lstm_8/strided_slice_1?
*model_8/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_8/lstm_8/TensorArrayV2/element_shape?
model_8/lstm_8/TensorArrayV2TensorListReserve3model_8/lstm_8/TensorArrayV2/element_shape:output:0'model_8/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_8/lstm_8/TensorArrayV2?
Dmodel_8/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2F
Dmodel_8/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
6model_8/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_8/lstm_8/transpose:y:0Mmodel_8/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_8/lstm_8/TensorArrayUnstack/TensorListFromTensor?
$model_8/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_8/lstm_8/strided_slice_2/stack?
&model_8/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_8/strided_slice_2/stack_1?
&model_8/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_8/strided_slice_2/stack_2?
model_8/lstm_8/strided_slice_2StridedSlicemodel_8/lstm_8/transpose:y:0-model_8/lstm_8/strided_slice_2/stack:output:0/model_8/lstm_8/strided_slice_2/stack_1:output:0/model_8/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2 
model_8/lstm_8/strided_slice_2?
0model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9model_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype022
0model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp?
!model_8/lstm_8/lstm_cell_8/MatMulMatMul'model_8/lstm_8/strided_slice_2:output:08model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_8/lstm_8/lstm_cell_8/MatMul?
2model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;model_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?
#model_8/lstm_8/lstm_cell_8/MatMul_1MatMulmodel_8/lstm_8/zeros:output:0:model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_8/lstm_8/lstm_cell_8/MatMul_1?
model_8/lstm_8/lstm_cell_8/addAddV2+model_8/lstm_8/lstm_cell_8/MatMul:product:0-model_8/lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
model_8/lstm_8/lstm_cell_8/add?
1model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:model_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?
"model_8/lstm_8/lstm_cell_8/BiasAddBiasAdd"model_8/lstm_8/lstm_cell_8/add:z:09model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_8/lstm_8/lstm_cell_8/BiasAdd?
 model_8/lstm_8/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_8/lstm_8/lstm_cell_8/Const?
*model_8/lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_8/lstm_8/lstm_cell_8/split/split_dim?
 model_8/lstm_8/lstm_cell_8/splitSplit3model_8/lstm_8/lstm_cell_8/split/split_dim:output:0+model_8/lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 model_8/lstm_8/lstm_cell_8/split?
"model_8/lstm_8/lstm_cell_8/SigmoidSigmoid)model_8/lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2$
"model_8/lstm_8/lstm_cell_8/Sigmoid?
$model_8/lstm_8/lstm_cell_8/Sigmoid_1Sigmoid)model_8/lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2&
$model_8/lstm_8/lstm_cell_8/Sigmoid_1?
model_8/lstm_8/lstm_cell_8/mulMul(model_8/lstm_8/lstm_cell_8/Sigmoid_1:y:0model_8/lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:??????????2 
model_8/lstm_8/lstm_cell_8/mul?
 model_8/lstm_8/lstm_cell_8/mul_1Mul&model_8/lstm_8/lstm_cell_8/Sigmoid:y:0)model_8/lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2"
 model_8/lstm_8/lstm_cell_8/mul_1?
 model_8/lstm_8/lstm_cell_8/add_1AddV2"model_8/lstm_8/lstm_cell_8/mul:z:0$model_8/lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_8/lstm_8/lstm_cell_8/add_1?
$model_8/lstm_8/lstm_cell_8/Sigmoid_2Sigmoid)model_8/lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2&
$model_8/lstm_8/lstm_cell_8/Sigmoid_2?
 model_8/lstm_8/lstm_cell_8/mul_2Mul(model_8/lstm_8/lstm_cell_8/Sigmoid_2:y:0$model_8/lstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_8/lstm_8/lstm_cell_8/mul_2?
,model_8/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2.
,model_8/lstm_8/TensorArrayV2_1/element_shape?
model_8/lstm_8/TensorArrayV2_1TensorListReserve5model_8/lstm_8/TensorArrayV2_1/element_shape:output:0'model_8/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_8/lstm_8/TensorArrayV2_1l
model_8/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_8/lstm_8/time?
'model_8/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_8/lstm_8/while/maximum_iterations?
!model_8/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_8/lstm_8/while/loop_counter?
model_8/lstm_8/whileWhile*model_8/lstm_8/while/loop_counter:output:00model_8/lstm_8/while/maximum_iterations:output:0model_8/lstm_8/time:output:0'model_8/lstm_8/TensorArrayV2_1:handle:0model_8/lstm_8/zeros:output:0model_8/lstm_8/zeros_1:output:0'model_8/lstm_8/strided_slice_1:output:0Fmodel_8/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource;model_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:model_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*,
body$R"
 model_8_lstm_8_while_body_165369*,
cond$R"
 model_8_lstm_8_while_cond_165368*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
model_8/lstm_8/while?
?model_8/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?model_8/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
1model_8/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStackmodel_8/lstm_8/while:output:3Hmodel_8/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype023
1model_8/lstm_8/TensorArrayV2Stack/TensorListStack?
$model_8/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$model_8/lstm_8/strided_slice_3/stack?
&model_8/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_8/lstm_8/strided_slice_3/stack_1?
&model_8/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_8/lstm_8/strided_slice_3/stack_2?
model_8/lstm_8/strided_slice_3StridedSlice:model_8/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-model_8/lstm_8/strided_slice_3/stack:output:0/model_8/lstm_8/strided_slice_3/stack_1:output:0/model_8/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2 
model_8/lstm_8/strided_slice_3?
model_8/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_8/lstm_8/transpose_1/perm?
model_8/lstm_8/transpose_1	Transpose:model_8/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0(model_8/lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
model_8/lstm_8/transpose_1?
model_8/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_8/lstm_8/runtime?
model_8/dropout_8/IdentityIdentity'model_8/lstm_8/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
model_8/dropout_8/Identity?
%model_8/dense_8/MatMul/ReadVariableOpReadVariableOp.model_8_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02'
%model_8/dense_8/MatMul/ReadVariableOp?
model_8/dense_8/MatMulMatMul#model_8/dropout_8/Identity:output:0-model_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_8/dense_8/MatMul?
&model_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_8/dense_8/BiasAdd/ReadVariableOp?
model_8/dense_8/BiasAddBiasAdd model_8/dense_8/MatMul:product:0.model_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_8/dense_8/BiasAdd?
model_8/dense_8/SigmoidSigmoid model_8/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
model_8/dense_8/Sigmoid?
IdentityIdentitymodel_8/dense_8/Sigmoid:y:0'^model_8/dense_8/BiasAdd/ReadVariableOp&^model_8/dense_8/MatMul/ReadVariableOp?^model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOpG^model_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp2^model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp1^model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp3^model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^model_8/lstm_8/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2P
&model_8/dense_8/BiasAdd/ReadVariableOp&model_8/dense_8/BiasAdd/ReadVariableOp2N
%model_8/dense_8/MatMul/ReadVariableOp%model_8/dense_8/MatMul/ReadVariableOp2?
>model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOp>model_8/fixed_adjacency_graph_convolution_8/add/ReadVariableOp2?
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp2?
Fmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOpFmodel_8/fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp2f
1model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp1model_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2d
0model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp0model_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp2h
2model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2model_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2,
model_8/lstm_8/whilemodel_8/lstm_8/while:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_17
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_166520

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
:
?????????F2
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_166437*
condR
while_cond_166436*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
b
F__inference_reshape_26_layer_call_and_return_conditional_losses_166215

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
:?????????
F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
F:W S
/
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_168083

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
while_body_165869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_8_165893_0
while_lstm_cell_8_165895_0
while_lstm_cell_8_165897_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_8_165893
while_lstm_cell_8_165895
while_lstm_cell_8_165897??)while/lstm_cell_8/StatefulPartitionedCall?
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
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_165893_0while_lstm_cell_8_165895_0while_lstm_cell_8_165897_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655442+
)while/lstm_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1*^while/lstm_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2*^while/lstm_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_165893while_lstm_cell_8_165893_0"6
while_lstm_cell_8_165895while_lstm_cell_8_165895_0"6
while_lstm_cell_8_165897while_lstm_cell_8_165897_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
C__inference_model_8_layer_call_and_return_conditional_losses_167255

inputsG
Cfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_8_add_readvariableop_resource5
1lstm_8_lstm_cell_8_matmul_readvariableop_resource7
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource6
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_8/add/ReadVariableOp?>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?(lstm_8/lstm_cell_8/MatMul/ReadVariableOp?*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?lstm_8/while?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinputs(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDimsx
reshape_24/ShapeShape$tf.expand_dims_8/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_24/Shape?
reshape_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_24/strided_slice/stack?
 reshape_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_1?
 reshape_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_24/strided_slice/stack_2?
reshape_24/strided_sliceStridedSlicereshape_24/Shape:output:0'reshape_24/strided_slice/stack:output:0)reshape_24/strided_slice/stack_1:output:0)reshape_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_24/strided_slicez
reshape_24/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_24/Reshape/shape/1z
reshape_24/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_24/Reshape/shape/2?
reshape_24/Reshape/shapePack!reshape_24/strided_slice:output:0#reshape_24/Reshape/shape/1:output:0#reshape_24/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_24/Reshape/shape?
reshape_24/ReshapeReshape$tf.expand_dims_8/ExpandDims:output:0!reshape_24/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_24/Reshape?
2fixed_adjacency_graph_convolution_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_8/transpose/perm?
-fixed_adjacency_graph_convolution_8/transpose	Transposereshape_24/Reshape:output:0;fixed_adjacency_graph_convolution_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_8/transpose?
)fixed_adjacency_graph_convolution_8/ShapeShape1fixed_adjacency_graph_convolution_8/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_8/Shape?
+fixed_adjacency_graph_convolution_8/unstackUnpack2fixed_adjacency_graph_convolution_8/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_8/unstack?
:fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_8/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_8/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_8/Shape_1?
-fixed_adjacency_graph_convolution_8/unstack_1Unpack4fixed_adjacency_graph_convolution_8/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_1?
1fixed_adjacency_graph_convolution_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_8/Reshape/shape?
+fixed_adjacency_graph_convolution_8/ReshapeReshape1fixed_adjacency_graph_convolution_8/transpose:y:0:fixed_adjacency_graph_convolution_8/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_8/Reshape?
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_8/transpose_1/perm?
/fixed_adjacency_graph_convolution_8/transpose_1	TransposeFfixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_8/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_8/transpose_1?
3fixed_adjacency_graph_convolution_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_8/Reshape_1/shape?
-fixed_adjacency_graph_convolution_8/Reshape_1Reshape3fixed_adjacency_graph_convolution_8/transpose_1:y:0<fixed_adjacency_graph_convolution_8/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_8/Reshape_1?
*fixed_adjacency_graph_convolution_8/MatMulMatMul4fixed_adjacency_graph_convolution_8/Reshape:output:06fixed_adjacency_graph_convolution_8/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_8/MatMul?
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_8/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_8/Reshape_2/shapePack4fixed_adjacency_graph_convolution_8/unstack:output:0>fixed_adjacency_graph_convolution_8/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_8/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_8/Reshape_2/shape?
-fixed_adjacency_graph_convolution_8/Reshape_2Reshape4fixed_adjacency_graph_convolution_8/MatMul:product:0<fixed_adjacency_graph_convolution_8/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_8/Reshape_2?
4fixed_adjacency_graph_convolution_8/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_8/transpose_2/perm?
/fixed_adjacency_graph_convolution_8/transpose_2	Transpose6fixed_adjacency_graph_convolution_8/Reshape_2:output:0=fixed_adjacency_graph_convolution_8/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_8/transpose_2?
+fixed_adjacency_graph_convolution_8/Shape_2Shape3fixed_adjacency_graph_convolution_8/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_8/Shape_2?
-fixed_adjacency_graph_convolution_8/unstack_2Unpack4fixed_adjacency_graph_convolution_8/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_2?
:fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02<
:fixed_adjacency_graph_convolution_8/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_8/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2-
+fixed_adjacency_graph_convolution_8/Shape_3?
-fixed_adjacency_graph_convolution_8/unstack_3Unpack4fixed_adjacency_graph_convolution_8/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_8/unstack_3?
3fixed_adjacency_graph_convolution_8/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_8/Reshape_3/shape?
-fixed_adjacency_graph_convolution_8/Reshape_3Reshape3fixed_adjacency_graph_convolution_8/transpose_2:y:0<fixed_adjacency_graph_convolution_8/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_8/Reshape_3?
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_8_shape_3_readvariableop_resource*
_output_shapes

:
*
dtype02@
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_8/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_8/transpose_3/perm?
/fixed_adjacency_graph_convolution_8/transpose_3	TransposeFfixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_8/transpose_3/perm:output:0*
T0*
_output_shapes

:
21
/fixed_adjacency_graph_convolution_8/transpose_3?
3fixed_adjacency_graph_convolution_8/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_8/Reshape_4/shape?
-fixed_adjacency_graph_convolution_8/Reshape_4Reshape3fixed_adjacency_graph_convolution_8/transpose_3:y:0<fixed_adjacency_graph_convolution_8/Reshape_4/shape:output:0*
T0*
_output_shapes

:
2/
-fixed_adjacency_graph_convolution_8/Reshape_4?
,fixed_adjacency_graph_convolution_8/MatMul_1MatMul6fixed_adjacency_graph_convolution_8/Reshape_3:output:06fixed_adjacency_graph_convolution_8/Reshape_4:output:0*
T0*'
_output_shapes
:?????????
2.
,fixed_adjacency_graph_convolution_8/MatMul_1?
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
27
5fixed_adjacency_graph_convolution_8/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_8/Reshape_5/shapePack6fixed_adjacency_graph_convolution_8/unstack_2:output:0>fixed_adjacency_graph_convolution_8/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_8/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_8/Reshape_5/shape?
-fixed_adjacency_graph_convolution_8/Reshape_5Reshape6fixed_adjacency_graph_convolution_8/MatMul_1:product:0<fixed_adjacency_graph_convolution_8/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F
2/
-fixed_adjacency_graph_convolution_8/Reshape_5?
6fixed_adjacency_graph_convolution_8/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_8_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_8/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_8/addAddV26fixed_adjacency_graph_convolution_8/Reshape_5:output:0>fixed_adjacency_graph_convolution_8/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F
2)
'fixed_adjacency_graph_convolution_8/add
reshape_25/ShapeShape+fixed_adjacency_graph_convolution_8/add:z:0*
T0*
_output_shapes
:2
reshape_25/Shape?
reshape_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_25/strided_slice/stack?
 reshape_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_25/strided_slice/stack_1?
 reshape_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_25/strided_slice/stack_2?
reshape_25/strided_sliceStridedSlicereshape_25/Shape:output:0'reshape_25/strided_slice/stack:output:0)reshape_25/strided_slice/stack_1:output:0)reshape_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_25/strided_slicez
reshape_25/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_25/Reshape/shape/1?
reshape_25/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_25/Reshape/shape/2z
reshape_25/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_25/Reshape/shape/3?
reshape_25/Reshape/shapePack!reshape_25/strided_slice:output:0#reshape_25/Reshape/shape/1:output:0#reshape_25/Reshape/shape/2:output:0#reshape_25/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_25/Reshape/shape?
reshape_25/ReshapeReshape+fixed_adjacency_graph_convolution_8/add:z:0!reshape_25/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F
2
reshape_25/Reshape?
permute_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_8/transpose/perm?
permute_8/transpose	Transposereshape_25/Reshape:output:0!permute_8/transpose/perm:output:0*
T0*/
_output_shapes
:?????????
F2
permute_8/transposek
reshape_26/ShapeShapepermute_8/transpose:y:0*
T0*
_output_shapes
:2
reshape_26/Shape?
reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_26/strided_slice/stack?
 reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_1?
 reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_2?
reshape_26/strided_sliceStridedSlicereshape_26/Shape:output:0'reshape_26/strided_slice/stack:output:0)reshape_26/strided_slice/stack_1:output:0)reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_26/strided_slice?
reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_26/Reshape/shape/1z
reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_26/Reshape/shape/2?
reshape_26/Reshape/shapePack!reshape_26/strided_slice:output:0#reshape_26/Reshape/shape/1:output:0#reshape_26/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_26/Reshape/shape?
reshape_26/ReshapeReshapepermute_8/transpose:y:0!reshape_26/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
F2
reshape_26/Reshapeg
lstm_8/ShapeShapereshape_26/Reshape:output:0*
T0*
_output_shapes
:2
lstm_8/Shape?
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack?
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1?
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2?
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicek
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/mul/y?
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/Less/y?
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessq
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/packed/1?
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const?
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/zeroso
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/mul/y?
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/Less/y?
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lessu
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/packed/1?
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const?
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/zeros_1?
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm?
lstm_8/transpose	Transposereshape_26/Reshape:output:0lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:
?????????F2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1?
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack?
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1?
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2?
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1?
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_8/TensorArrayV2/element_shape?
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2?
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor?
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack?
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1?
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2?
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_8/strided_slice_2?
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp?
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/MatMul?
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/MatMul_1?
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/add?
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/BiasAddv
lstm_8/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/lstm_cell_8/Const?
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dim?
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_8/lstm_cell_8/split?
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid?
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid_1?
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul?
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0!lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul_1?
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/add_1?
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/Sigmoid_2?
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0lstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_8/mul_2?
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$lstm_8/TensorArrayV2_1/element_shape?
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time?
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counter?
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_8_while_body_167164*$
condR
lstm_8_while_cond_167163*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_8/while?
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack?
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_8/strided_slice_3/stack?
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1?
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2?
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_8/strided_slice_3?
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm?
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtime?
dropout_8/IdentityIdentitylstm_8/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
dropout_8/Identity?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_8/BiasAddy
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_8/Sigmoid?
IdentityIdentitydense_8/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_8/add/ReadVariableOp?^fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_8/add/ReadVariableOp6fixed_adjacency_graph_convolution_8/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_8/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_8/transpose_3/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
 model_8_lstm_8_while_cond_165368:
6model_8_lstm_8_while_model_8_lstm_8_while_loop_counter@
<model_8_lstm_8_while_model_8_lstm_8_while_maximum_iterations$
 model_8_lstm_8_while_placeholder&
"model_8_lstm_8_while_placeholder_1&
"model_8_lstm_8_while_placeholder_2&
"model_8_lstm_8_while_placeholder_3<
8model_8_lstm_8_while_less_model_8_lstm_8_strided_slice_1R
Nmodel_8_lstm_8_while_model_8_lstm_8_while_cond_165368___redundant_placeholder0R
Nmodel_8_lstm_8_while_model_8_lstm_8_while_cond_165368___redundant_placeholder1R
Nmodel_8_lstm_8_while_model_8_lstm_8_while_cond_165368___redundant_placeholder2R
Nmodel_8_lstm_8_while_model_8_lstm_8_while_cond_165368___redundant_placeholder3!
model_8_lstm_8_while_identity
?
model_8/lstm_8/while/LessLess model_8_lstm_8_while_placeholder8model_8_lstm_8_while_less_model_8_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
model_8/lstm_8/while/Less?
model_8/lstm_8/while/IdentityIdentitymodel_8/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
model_8/lstm_8/while/Identity"G
model_8_lstm_8_while_identity&model_8/lstm_8/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168165

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
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_167630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167630___redundant_placeholder04
0while_while_cond_167630___redundant_placeholder14
0while_while_cond_167630___redundant_placeholder24
0while_while_cond_167630___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
ʇ
?
"__inference__traced_restore_168418
file_prefix:
6assignvariableop_fixed_adjacency_graph_convolution_8_aA
=assignvariableop_1_fixed_adjacency_graph_convolution_8_kernel?
;assignvariableop_2_fixed_adjacency_graph_convolution_8_bias%
!assignvariableop_3_dense_8_kernel#
assignvariableop_4_dense_8_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate1
-assignvariableop_10_lstm_8_lstm_cell_8_kernel;
7assignvariableop_11_lstm_8_lstm_cell_8_recurrent_kernel/
+assignvariableop_12_lstm_8_lstm_cell_8_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1I
Eassignvariableop_17_adam_fixed_adjacency_graph_convolution_8_kernel_mG
Cassignvariableop_18_adam_fixed_adjacency_graph_convolution_8_bias_m-
)assignvariableop_19_adam_dense_8_kernel_m+
'assignvariableop_20_adam_dense_8_bias_m8
4assignvariableop_21_adam_lstm_8_lstm_cell_8_kernel_mB
>assignvariableop_22_adam_lstm_8_lstm_cell_8_recurrent_kernel_m6
2assignvariableop_23_adam_lstm_8_lstm_cell_8_bias_mI
Eassignvariableop_24_adam_fixed_adjacency_graph_convolution_8_kernel_vG
Cassignvariableop_25_adam_fixed_adjacency_graph_convolution_8_bias_v-
)assignvariableop_26_adam_dense_8_kernel_v+
'assignvariableop_27_adam_dense_8_bias_v8
4assignvariableop_28_adam_lstm_8_lstm_cell_8_kernel_vB
>assignvariableop_29_adam_lstm_8_lstm_cell_8_recurrent_kernel_v6
2assignvariableop_30_adam_lstm_8_lstm_cell_8_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp6assignvariableop_fixed_adjacency_graph_convolution_8_aIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp=assignvariableop_1_fixed_adjacency_graph_convolution_8_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp;assignvariableop_2_fixed_adjacency_graph_convolution_8_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_8_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_8_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_8_lstm_cell_8_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_8_lstm_cell_8_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_8_lstm_cell_8_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_adam_fixed_adjacency_graph_convolution_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpCassignvariableop_18_adam_fixed_adjacency_graph_convolution_8_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_8_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_8_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_8_lstm_cell_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_lstm_8_lstm_cell_8_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_lstm_8_lstm_cell_8_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpEassignvariableop_24_adam_fixed_adjacency_graph_convolution_8_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpCassignvariableop_25_adam_fixed_adjacency_graph_convolution_8_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_8_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_8_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_8_lstm_cell_8_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_lstm_8_lstm_cell_8_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_lstm_8_lstm_cell_8_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*?
_input_shapes?
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
?
b
F__inference_reshape_24_layer_call_and_return_conditional_losses_167310

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
?J
?	
lstm_8_while_body_166919*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0?
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0>
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor;
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource=
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource<
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource??/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem?
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_8/while/lstm_cell_8/MatMul?
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_8/while/lstm_cell_8/MatMul_1?
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_8/add?
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_8/BiasAdd?
lstm_8/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_8/while/lstm_cell_8/Const?
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dim?
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2 
lstm_8/while/lstm_cell_8/split?
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_8/Sigmoid?
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_8/Sigmoid_1?
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_8/mul?
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0'lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/mul_1?
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/add_1?
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_8/Sigmoid_2?
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/mul_2?
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y?
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y?
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1?
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity?
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1?
lstm_8/while/Identity_2Identitylstm_8/while/add:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2?
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3?
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_8/while/Identity_4?
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_8/while/Identity_5"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"?
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_fixed_adjacency_graph_convolution_8_layer_call_fn_167379
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
:?????????F
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_1661592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????F
2

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
?@
?
while_body_167951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168134

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
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?@
?
while_body_166288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_reshape_24_layer_call_fn_167315

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
F__inference_reshape_24_layer_call_and_return_conditional_losses_1660982
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
?
F
*__inference_permute_8_layer_call_fn_165473

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
GPU 2J 8? *N
fIRG
E__inference_permute_8_layer_call_and_return_conditional_losses_1654672
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
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_166567

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_168073

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
C__inference_model_8_layer_call_and_return_conditional_losses_166671

inputs.
*fixed_adjacency_graph_convolution_8_166647.
*fixed_adjacency_graph_convolution_8_166649.
*fixed_adjacency_graph_convolution_8_166651
lstm_8_166657
lstm_8_166659
lstm_8_166661
dense_8_166665
dense_8_166667
identity??dense_8/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinputs(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDims?
reshape_24/PartitionedCallPartitionedCall$tf.expand_dims_8/ExpandDims:output:0*
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
F__inference_reshape_24_layer_call_and_return_conditional_losses_1660982
reshape_24/PartitionedCall?
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_24/PartitionedCall:output:0*fixed_adjacency_graph_convolution_8_166647*fixed_adjacency_graph_convolution_8_166649*fixed_adjacency_graph_convolution_8_166651*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_1661592=
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?
reshape_25/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_25_layer_call_and_return_conditional_losses_1661932
reshape_25/PartitionedCall?
permute_8/PartitionedCallPartitionedCall#reshape_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_permute_8_layer_call_and_return_conditional_losses_1654672
permute_8/PartitionedCall?
reshape_26/PartitionedCallPartitionedCall"permute_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_1662152
reshape_26/PartitionedCall?
lstm_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:0lstm_8_166657lstm_8_166659lstm_8_166661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1663712 
lstm_8/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665622#
!dropout_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_166665dense_8_166667*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1665912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_8/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?,
?
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_166159
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

:
*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
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

:
*
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

:
2
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

:
2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:?????????
2

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
value	B :
2
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
:?????????F
2
	Reshape_5?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F
2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:?????????F
2

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
?,
?
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_167368
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

:
*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   2	
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

:
*
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

:
2
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

:
2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:?????????
2

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
value	B :
2
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
:?????????F
2
	Reshape_5?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:F*
dtype02
add/ReadVariableOpy
addAddV2Reshape_5:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F
2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:?????????F
2

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
?
G
+__inference_reshape_25_layer_call_fn_167398

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
:?????????F
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_25_layer_call_and_return_conditional_losses_1661932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????F
2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F
:S O
+
_output_shapes
:?????????F

 
_user_specified_nameinputs
?
?
while_cond_165868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_165868___redundant_placeholder04
0while_while_cond_165868___redundant_placeholder14
0while_while_cond_165868___redundant_placeholder24
0while_while_cond_165868___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
lstm_8_while_cond_166918*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_166918___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_166918___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_166918___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_166918___redundant_placeholder3
lstm_8_while_identity
?
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?J
?	
lstm_8_while_body_167164*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0?
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0>
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor;
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource=
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource<
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource??/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem?
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp?
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_8/while/lstm_cell_8/MatMul?
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp?
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_8/while/lstm_cell_8/MatMul_1?
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_8/add?
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp?
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_8/BiasAdd?
lstm_8/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_8/while/lstm_cell_8/Const?
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dim?
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2 
lstm_8/while/lstm_cell_8/split?
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_8/Sigmoid?
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_8/Sigmoid_1?
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_8/mul?
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0'lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/mul_1?
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/add_1?
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_8/Sigmoid_2?
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_8/while/lstm_cell_8/mul_2?
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y?
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y?
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1?
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity?
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1?
lstm_8/while/Identity_2Identitylstm_8/while/add:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2?
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3?
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_8/while/Identity_4?
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:00^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_8/while/Identity_5"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"?
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_model_8_layer_call_fn_166741
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_1667222
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
input_17
?
?
'__inference_lstm_8_layer_call_fn_168045

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1663712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
?
(__inference_model_8_layer_call_fn_167276

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
GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_1666712
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
while_cond_166000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_166000___redundant_placeholder04
0while_while_cond_166000___redundant_placeholder14
0while_while_cond_166000___redundant_placeholder24
0while_while_cond_166000___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_167631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167885

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
:
?????????F2
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_167802*
condR
while_cond_167801*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_165544

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
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
b
F__inference_reshape_26_layer_call_and_return_conditional_losses_167411

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
:?????????
F2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
F:W S
/
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?@
?
while_body_167802
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
C__inference_dense_8_layer_call_and_return_conditional_losses_168094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?F*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_8_layer_call_and_return_conditional_losses_166591

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?F*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_reshape_25_layer_call_and_return_conditional_losses_166193

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
:?????????F
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????F
2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F
:S O
+
_output_shapes
:?????????F

 
_user_specified_nameinputs
?%
?
while_body_166001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_8_166025_0
while_lstm_cell_8_166027_0
while_lstm_cell_8_166029_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_8_166025
while_lstm_cell_8_166027
while_lstm_cell_8_166029??)while/lstm_cell_8/StatefulPartitionedCall?
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
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_166025_0while_lstm_cell_8_166027_0while_lstm_cell_8_166029_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655752+
)while/lstm_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1*^while/lstm_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2*^while/lstm_cell_8/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_166025while_lstm_cell_8_166025_0"6
while_lstm_cell_8_166027while_lstm_cell_8_166027_0"6
while_lstm_cell_8_166029while_lstm_cell_8_166029_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_reshape_26_layer_call_fn_167416

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
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_1662152
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????
F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
F:W S
/
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
?
(__inference_model_8_layer_call_fn_166690
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_1666712
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
input_17
?
?
,__inference_lstm_cell_8_layer_call_fn_168182

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
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?&
?
C__inference_model_8_layer_call_and_return_conditional_losses_166638
input_17.
*fixed_adjacency_graph_convolution_8_166614.
*fixed_adjacency_graph_convolution_8_166616.
*fixed_adjacency_graph_convolution_8_166618
lstm_8_166624
lstm_8_166626
lstm_8_166628
dense_8_166632
dense_8_166634
identity??dense_8/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinput_17(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDims?
reshape_24/PartitionedCallPartitionedCall$tf.expand_dims_8/ExpandDims:output:0*
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
F__inference_reshape_24_layer_call_and_return_conditional_losses_1660982
reshape_24/PartitionedCall?
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_24/PartitionedCall:output:0*fixed_adjacency_graph_convolution_8_166614*fixed_adjacency_graph_convolution_8_166616*fixed_adjacency_graph_convolution_8_166618*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_1661592=
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?
reshape_25/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_25_layer_call_and_return_conditional_losses_1661932
reshape_25/PartitionedCall?
permute_8/PartitionedCallPartitionedCall#reshape_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_permute_8_layer_call_and_return_conditional_losses_1654672
permute_8/PartitionedCall?
reshape_26/PartitionedCallPartitionedCall"permute_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_1662152
reshape_26/PartitionedCall?
lstm_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:0lstm_8_166624lstm_8_166626lstm_8_166628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1665202 
lstm_8/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665672
dropout_8/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_166632dense_8_166634*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1665912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_8/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_17
?Y
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167714
inputs_0.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_167631*
condR
while_cond_167630*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?H
?
__inference__traced_save_168315
file_prefixD
@savev2_fixed_adjacency_graph_convolution_8_a_read_readvariableopI
Esavev2_fixed_adjacency_graph_convolution_8_kernel_read_readvariableopG
Csavev2_fixed_adjacency_graph_convolution_8_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_8_kernel_m_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_8_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_8_kernel_v_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_8_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B1layer_with_weights-0/A/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_fixed_adjacency_graph_convolution_8_a_read_readvariableopEsavev2_fixed_adjacency_graph_convolution_8_kernel_read_readvariableopCsavev2_fixed_adjacency_graph_convolution_8_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_8_kernel_m_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_8_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_8_kernel_v_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_8_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :FF:
:F:	?F:F: : : : : :	F?:
??:?: : : : :
:F:	?F:F:	F?:
??:?:
:F:	?F:F:	F?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:FF:$ 

_output_shapes

:
:$ 

_output_shapes

:F:%!

_output_shapes
:	?F: 
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
:	F?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:
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

:
:$ 

_output_shapes

:F:%!

_output_shapes
:	?F: 

_output_shapes
:F:%!

_output_shapes
:	F?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:$ 

_output_shapes

:
:$ 

_output_shapes

:F:%!

_output_shapes
:	?F: 

_output_shapes
:F:%!

_output_shapes
:	F?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?: 

_output_shapes
: 
?
?
while_cond_167801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167801___redundant_placeholder04
0while_while_cond_167801___redundant_placeholder14
0while_while_cond_167801___redundant_placeholder24
0while_while_cond_167801___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?(
?
C__inference_model_8_layer_call_and_return_conditional_losses_166608
input_17.
*fixed_adjacency_graph_convolution_8_166172.
*fixed_adjacency_graph_convolution_8_166174.
*fixed_adjacency_graph_convolution_8_166176
lstm_8_166543
lstm_8_166545
lstm_8_166547
dense_8_166602
dense_8_166604
identity??dense_8/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinput_17(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDims?
reshape_24/PartitionedCallPartitionedCall$tf.expand_dims_8/ExpandDims:output:0*
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
F__inference_reshape_24_layer_call_and_return_conditional_losses_1660982
reshape_24/PartitionedCall?
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_24/PartitionedCall:output:0*fixed_adjacency_graph_convolution_8_166172*fixed_adjacency_graph_convolution_8_166174*fixed_adjacency_graph_convolution_8_166176*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_1661592=
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?
reshape_25/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_25_layer_call_and_return_conditional_losses_1661932
reshape_25/PartitionedCall?
permute_8/PartitionedCallPartitionedCall#reshape_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_permute_8_layer_call_and_return_conditional_losses_1654672
permute_8/PartitionedCall?
reshape_26/PartitionedCallPartitionedCall"permute_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_1662152
reshape_26/PartitionedCall?
lstm_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:0lstm_8_166543lstm_8_166545lstm_8_166547*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1663712 
lstm_8/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665622#
!dropout_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_166602dense_8_166604*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1665912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_8/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_17
?&
?
C__inference_model_8_layer_call_and_return_conditional_losses_166722

inputs.
*fixed_adjacency_graph_convolution_8_166698.
*fixed_adjacency_graph_convolution_8_166700.
*fixed_adjacency_graph_convolution_8_166702
lstm_8_166708
lstm_8_166710
lstm_8_166712
dense_8_166716
dense_8_166718
identity??dense_8/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimsinputs(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_8/ExpandDims?
reshape_24/PartitionedCallPartitionedCall$tf.expand_dims_8/ExpandDims:output:0*
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
F__inference_reshape_24_layer_call_and_return_conditional_losses_1660982
reshape_24/PartitionedCall?
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_24/PartitionedCall:output:0*fixed_adjacency_graph_convolution_8_166698*fixed_adjacency_graph_convolution_8_166700*fixed_adjacency_graph_convolution_8_166702*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????F
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_1661592=
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall?
reshape_25/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????F
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_25_layer_call_and_return_conditional_losses_1661932
reshape_25/PartitionedCall?
permute_8/PartitionedCallPartitionedCall#reshape_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_permute_8_layer_call_and_return_conditional_losses_1654672
permute_8/PartitionedCall?
reshape_26/PartitionedCallPartitionedCall"permute_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
F* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_1662152
reshape_26/PartitionedCall?
lstm_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:0lstm_8_166708lstm_8_166710lstm_8_166712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1665202 
lstm_8/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_1665672
dropout_8/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_166716dense_8_166718*
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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1665912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_8/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall;fixed_adjacency_graph_convolution_8/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
'__inference_lstm_8_layer_call_fn_167736
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
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_8_layer_call_and_return_conditional_losses_1660702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
b
F__inference_reshape_24_layer_call_and_return_conditional_losses_166098

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
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_166371

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
:
?????????F2
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_166288*
condR
while_cond_166287*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_166772
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_1654602
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
input_17
?
?
(__inference_model_8_layer_call_fn_167297

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
GPU 2J 8? *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_1667222
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
?D
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_166070

inputs
lstm_cell_8_165988
lstm_cell_8_165990
lstm_cell_8_165992
identity??#lstm_cell_8/StatefulPartitionedCall?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_165988lstm_cell_8_165990lstm_cell_8_165992*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655752%
#lstm_cell_8/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_165988lstm_cell_8_165990lstm_cell_8_165992*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_166001*
condR
while_cond_166000*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_8/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_168034

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
:
?????????F2
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_167951*
condR
while_cond_167950*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
??????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????
?2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
F
 
_user_specified_nameinputs
?
?
while_cond_166436
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_166436___redundant_placeholder04
0while_while_cond_166436___redundant_placeholder14
0while_while_cond_166436___redundant_placeholder24
0while_while_cond_166436___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_167950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167950___redundant_placeholder04
0while_while_cond_167950___redundant_placeholder14
0while_while_cond_167950___redundant_placeholder24
0while_while_cond_167950___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_165575

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
:	F?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mule
mul_1MulSigmoid:y:0split:output:2*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2b
mul_2MulSigmoid_2:y:0	add_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????F:??????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?@
?
while_body_166437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??(while/lstm_cell_8/BiasAdd/ReadVariableOp?'while/lstm_cell_8/MatMul/ReadVariableOp?)while/lstm_cell_8/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?D
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_165938

inputs
lstm_cell_8_165856
lstm_cell_8_165858
lstm_cell_8_165860
identity??#lstm_cell_8/StatefulPartitionedCall?whileD
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_165856lstm_cell_8_165858lstm_cell_8_165860*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1655442%
#lstm_cell_8/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_165856lstm_cell_8_165858lstm_cell_8_165860*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_165869*
condR
while_cond_165868*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_8/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?Y
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167565
inputs_0.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??"lstm_cell_8/BiasAdd/ReadVariableOp?!lstm_cell_8/MatMul/ReadVariableOp?#lstm_cell_8/MatMul_1/ReadVariableOp?whileF
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
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
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
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
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_8/Sigmoid_2?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_167482*
condR
while_cond_167481*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?
?
while_cond_167481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_167481___redundant_placeholder04
0while_while_cond_167481___redundant_placeholder14
0while_while_cond_167481___redundant_placeholder24
0while_while_cond_167481___redundant_placeholder3
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
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
b
F__inference_reshape_25_layer_call_and_return_conditional_losses_167393

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
:?????????F
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????F
2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????F
:S O
+
_output_shapes
:?????????F

 
_user_specified_nameinputs
?
a
E__inference_permute_8_layer_call_and_return_conditional_losses_165467

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
}
(__inference_dense_8_layer_call_fn_168103

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
GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_1665912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
lstm_8_while_cond_167163*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_167163___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_167163___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_167163___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_167163___redundant_placeholder3
lstm_8_while_identity
?
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 
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
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input_175
serving_default_input_17:0?????????F;
dense_80
StatefulPartitionedCall:0?????????Ftensorflow/serving/predict:??
?C
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
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?@
_tf_keras_network??{"class_name": "Functional", "name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}, "name": "input_17", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["input_17", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_24", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_24", "inbound_nodes": [[["tf.expand_dims_8", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_8", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_8", "inbound_nodes": [[["reshape_24", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_25", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_25", "inbound_nodes": [[["fixed_adjacency_graph_convolution_8", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_8", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_8", "inbound_nodes": [[["reshape_25", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_26", "inbound_nodes": [[["permute_8", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_8", "inbound_nodes": [[["reshape_26", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["lstm_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}], "input_layers": [["input_17", 0, 0]], "output_layers": [["dense_8", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}, "name": "input_17", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["input_17", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_24", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_24", "inbound_nodes": [[["tf.expand_dims_8", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_8", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_8", "inbound_nodes": [[["reshape_24", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_25", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_25", "inbound_nodes": [[["fixed_adjacency_graph_convolution_8", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_8", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_8", "inbound_nodes": [[["reshape_25", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_26", "inbound_nodes": [[["permute_8", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_8", "inbound_nodes": [[["reshape_26", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["lstm_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}], "input_layers": [["input_17", 0, 0]], "output_layers": [["dense_8", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_17", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}
?
	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_24", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
?
A

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "FixedAdjacencyGraphConvolution", "name": "fixed_adjacency_graph_convolution_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_adjacency_graph_convolution_8", "trainable": true, "dtype": "float32", "units": 10, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}}
?
regularization_losses
trainable_variables
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_25", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}}
?
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Permute", "name": "permute_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "permute_8", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}}
?
)cell
*
state_spec
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 70]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 70]}}
?
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
9iter

:beta_1

;beta_2
	<decay
=learning_ratem?m?3m?4m?>m??m?@m?v?v?3v?4v?>v??v?@v?"
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
?
regularization_losses

Alayers
trainable_variables
Bnon_trainable_variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

Flayers
trainable_variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3FF2%fixed_adjacency_graph_convolution_8/A
<::
2*fixed_adjacency_graph_convolution_8/kernel
::8F2(fixed_adjacency_graph_convolution_8/bias
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
?
regularization_losses

Klayers
trainable_variables
Lnon_trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
	variables
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
regularization_losses

Players
trainable_variables
Qnon_trainable_variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
	variables
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
!regularization_losses

Ulayers
"trainable_variables
Vnon_trainable_variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
#	variables
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
%regularization_losses

Zlayers
&trainable_variables
[non_trainable_variables
\layer_regularization_losses
]metrics
^layer_metrics
'	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

>kernel
?recurrent_kernel
@bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_8", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
?
+regularization_losses

clayers
,trainable_variables
dnon_trainable_variables
elayer_regularization_losses
fmetrics
glayer_metrics
-	variables

hstates
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
/regularization_losses

ilayers
0trainable_variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
1	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?F2dense_8/kernel
:F2dense_8/bias
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
?
5regularization_losses

nlayers
6trainable_variables
onon_trainable_variables
player_regularization_losses
qmetrics
rlayer_metrics
7	variables
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
,:*	F?2lstm_8/lstm_cell_8/kernel
7:5
??2#lstm_8/lstm_cell_8/recurrent_kernel
&:$?2lstm_8/lstm_cell_8/bias
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
?
_regularization_losses

ulayers
`trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
xmetrics
ylayer_metrics
a	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
	ztotal
	{count
|	variables
}	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	~total
	count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
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
?	variables"
_generic_user_object
A:?
21Adam/fixed_adjacency_graph_convolution_8/kernel/m
?:=F2/Adam/fixed_adjacency_graph_convolution_8/bias/m
&:$	?F2Adam/dense_8/kernel/m
:F2Adam/dense_8/bias/m
1:/	F?2 Adam/lstm_8/lstm_cell_8/kernel/m
<::
??2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
+:)?2Adam/lstm_8/lstm_cell_8/bias/m
A:?
21Adam/fixed_adjacency_graph_convolution_8/kernel/v
?:=F2/Adam/fixed_adjacency_graph_convolution_8/bias/v
&:$	?F2Adam/dense_8/kernel/v
:F2Adam/dense_8/bias/v
1:/	F?2 Adam/lstm_8/lstm_cell_8/kernel/v
<::
??2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
+:)?2Adam/lstm_8/lstm_cell_8/bias/v
?2?
C__inference_model_8_layer_call_and_return_conditional_losses_166638
C__inference_model_8_layer_call_and_return_conditional_losses_167017
C__inference_model_8_layer_call_and_return_conditional_losses_167255
C__inference_model_8_layer_call_and_return_conditional_losses_166608?
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
!__inference__wrapped_model_165460?
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
annotations? *+?(
&?#
input_17?????????F
?2?
(__inference_model_8_layer_call_fn_166741
(__inference_model_8_layer_call_fn_167276
(__inference_model_8_layer_call_fn_167297
(__inference_model_8_layer_call_fn_166690?
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
F__inference_reshape_24_layer_call_and_return_conditional_losses_167310?
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
+__inference_reshape_24_layer_call_fn_167315?
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
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_167368?
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
D__inference_fixed_adjacency_graph_convolution_8_layer_call_fn_167379?
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
F__inference_reshape_25_layer_call_and_return_conditional_losses_167393?
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
+__inference_reshape_25_layer_call_fn_167398?
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
E__inference_permute_8_layer_call_and_return_conditional_losses_165467?
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
*__inference_permute_8_layer_call_fn_165473?
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
F__inference_reshape_26_layer_call_and_return_conditional_losses_167411?
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
+__inference_reshape_26_layer_call_fn_167416?
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_167714
B__inference_lstm_8_layer_call_and_return_conditional_losses_167885
B__inference_lstm_8_layer_call_and_return_conditional_losses_167565
B__inference_lstm_8_layer_call_and_return_conditional_losses_168034?
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
'__inference_lstm_8_layer_call_fn_167725
'__inference_lstm_8_layer_call_fn_168045
'__inference_lstm_8_layer_call_fn_167736
'__inference_lstm_8_layer_call_fn_168056?
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_168068
E__inference_dropout_8_layer_call_and_return_conditional_losses_168073?
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
*__inference_dropout_8_layer_call_fn_168078
*__inference_dropout_8_layer_call_fn_168083?
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
C__inference_dense_8_layer_call_and_return_conditional_losses_168094?
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
(__inference_dense_8_layer_call_fn_168103?
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
?B?
$__inference_signature_wrapper_166772input_17"?
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
?2?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168134
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168165?
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
,__inference_lstm_cell_8_layer_call_fn_168182
,__inference_lstm_cell_8_layer_call_fn_168199?
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
!__inference__wrapped_model_165460t>?@345?2
+?(
&?#
input_17?????????F
? "1?.
,
dense_8!?
dense_8?????????F?
C__inference_dense_8_layer_call_and_return_conditional_losses_168094]340?-
&?#
!?
inputs??????????
? "%?"
?
0?????????F
? |
(__inference_dense_8_layer_call_fn_168103P340?-
&?#
!?
inputs??????????
? "??????????F?
E__inference_dropout_8_layer_call_and_return_conditional_losses_168068^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_168073^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? 
*__inference_dropout_8_layer_call_fn_168078Q4?1
*?'
!?
inputs??????????
p
? "???????????
*__inference_dropout_8_layer_call_fn_168083Q4?1
*?'
!?
inputs??????????
p 
? "????????????
___inference_fixed_adjacency_graph_convolution_8_layer_call_and_return_conditional_losses_167368g5?2
+?(
&?#
features?????????F
? ")?&
?
0?????????F

? ?
D__inference_fixed_adjacency_graph_convolution_8_layer_call_fn_167379Z5?2
+?(
&?#
features?????????F
? "??????????F
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167565~>?@O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p

 
? "&?#
?
0??????????
? ?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167714~>?@O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p 

 
? "&?#
?
0??????????
? ?
B__inference_lstm_8_layer_call_and_return_conditional_losses_167885n>?@??<
5?2
$?!
inputs?????????
F

 
p

 
? "&?#
?
0??????????
? ?
B__inference_lstm_8_layer_call_and_return_conditional_losses_168034n>?@??<
5?2
$?!
inputs?????????
F

 
p 

 
? "&?#
?
0??????????
? ?
'__inference_lstm_8_layer_call_fn_167725q>?@O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p

 
? "????????????
'__inference_lstm_8_layer_call_fn_167736q>?@O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p 

 
? "????????????
'__inference_lstm_8_layer_call_fn_168045a>?@??<
5?2
$?!
inputs?????????
F

 
p

 
? "????????????
'__inference_lstm_8_layer_call_fn_168056a>?@??<
5?2
$?!
inputs?????????
F

 
p 

 
? "????????????
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168134?>?@??
x?u
 ?
inputs?????????F
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_168165?>?@??
x?u
 ?
inputs?????????F
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
,__inference_lstm_cell_8_layer_call_fn_168182?>?@??
x?u
 ?
inputs?????????F
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
,__inference_lstm_cell_8_layer_call_fn_168199?>?@??
x?u
 ?
inputs?????????F
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
C__inference_model_8_layer_call_and_return_conditional_losses_166608p>?@34=?:
3?0
&?#
input_17?????????F
p

 
? "%?"
?
0?????????F
? ?
C__inference_model_8_layer_call_and_return_conditional_losses_166638p>?@34=?:
3?0
&?#
input_17?????????F
p 

 
? "%?"
?
0?????????F
? ?
C__inference_model_8_layer_call_and_return_conditional_losses_167017n>?@34;?8
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
C__inference_model_8_layer_call_and_return_conditional_losses_167255n>?@34;?8
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
(__inference_model_8_layer_call_fn_166690c>?@34=?:
3?0
&?#
input_17?????????F
p

 
? "??????????F?
(__inference_model_8_layer_call_fn_166741c>?@34=?:
3?0
&?#
input_17?????????F
p 

 
? "??????????F?
(__inference_model_8_layer_call_fn_167276a>?@34;?8
1?.
$?!
inputs?????????F
p

 
? "??????????F?
(__inference_model_8_layer_call_fn_167297a>?@34;?8
1?.
$?!
inputs?????????F
p 

 
? "??????????F?
E__inference_permute_8_layer_call_and_return_conditional_losses_165467?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_permute_8_layer_call_fn_165473?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_reshape_24_layer_call_and_return_conditional_losses_167310d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
+__inference_reshape_24_layer_call_fn_167315W7?4
-?*
(?%
inputs?????????F
? "??????????F?
F__inference_reshape_25_layer_call_and_return_conditional_losses_167393d3?0
)?&
$?!
inputs?????????F

? "-?*
#? 
0?????????F

? ?
+__inference_reshape_25_layer_call_fn_167398W3?0
)?&
$?!
inputs?????????F

? " ??????????F
?
F__inference_reshape_26_layer_call_and_return_conditional_losses_167411d7?4
-?*
(?%
inputs?????????
F
? ")?&
?
0?????????
F
? ?
+__inference_reshape_26_layer_call_fn_167416W7?4
-?*
(?%
inputs?????????
F
? "??????????
F?
$__inference_signature_wrapper_166772?>?@34A?>
? 
7?4
2
input_17&?#
input_17?????????F"1?.
,
dense_8!?
dense_8?????????F