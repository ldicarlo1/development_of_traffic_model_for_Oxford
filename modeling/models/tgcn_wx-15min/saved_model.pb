??&
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
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??"
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
*fixed_adjacency_graph_convolution_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*fixed_adjacency_graph_convolution_2/kernel
?
>fixed_adjacency_graph_convolution_2/kernel/Read/ReadVariableOpReadVariableOp*fixed_adjacency_graph_convolution_2/kernel*
_output_shapes

:*
dtype0
?
(fixed_adjacency_graph_convolution_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*9
shared_name*(fixed_adjacency_graph_convolution_2/bias
?
<fixed_adjacency_graph_convolution_2/bias/Read/ReadVariableOpReadVariableOp(fixed_adjacency_graph_convolution_2/bias*
_output_shapes

:F*
dtype0
?
lstm_2/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?**
shared_namelstm_2/lstm_cell_2/kernel
?
-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/kernel*
_output_shapes
:	F?*
dtype0
?
#lstm_2/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_2/lstm_cell_2/recurrent_kernel
?
7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_2/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_2/lstm_cell_2/bias
?
+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/bias*
_output_shapes	
:?*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?F*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:F*
dtype0
?
%fixed_adjacency_graph_convolution_2/AVarHandleOp*
_output_shapes
: *
dtype0*
shape
:FF*6
shared_name'%fixed_adjacency_graph_convolution_2/A
?
9fixed_adjacency_graph_convolution_2/A/Read/ReadVariableOpReadVariableOp%fixed_adjacency_graph_convolution_2/A*
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
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
1Adam/fixed_adjacency_graph_convolution_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/fixed_adjacency_graph_convolution_2/kernel/m
?
EAdam/fixed_adjacency_graph_convolution_2/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_2/kernel/m*
_output_shapes

:*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_2/bias/m
?
CAdam/fixed_adjacency_graph_convolution_2/bias/m/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_2/bias/m*
_output_shapes

:F*
dtype0
?
 Adam/lstm_2/lstm_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/m
?
4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/m*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
?
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_2/bias/m
?
2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	?F*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:F*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
?
1Adam/fixed_adjacency_graph_convolution_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/fixed_adjacency_graph_convolution_2/kernel/v
?
EAdam/fixed_adjacency_graph_convolution_2/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/fixed_adjacency_graph_convolution_2/kernel/v*
_output_shapes

:*
dtype0
?
/Adam/fixed_adjacency_graph_convolution_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*@
shared_name1/Adam/fixed_adjacency_graph_convolution_2/bias/v
?
CAdam/fixed_adjacency_graph_convolution_2/bias/v/Read/ReadVariableOpReadVariableOp/Adam/fixed_adjacency_graph_convolution_2/bias/v*
_output_shapes

:F*
dtype0
?
 Adam/lstm_2/lstm_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	F?*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/v
?
4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/v*
_output_shapes
:	F?*
dtype0
?
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
?
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_2/bias/v
?
2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?F*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	?F*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
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
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE*fixed_adjacency_graph_convolution_2/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(fixed_adjacency_graph_convolution_2/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_2/lstm_cell_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_2/lstm_cell_2/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_2/lstm_cell_2/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_6/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_6/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%fixed_adjacency_graph_convolution_2/A&variables/4/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_2/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_2/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_6/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_6/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/fixed_adjacency_graph_convolution_2/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/fixed_adjacency_graph_convolution_2/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_6/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_6/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_13Placeholder*/
_output_shapes
:?????????F*
dtype0*$
shape:?????????F
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13dense_7/kerneldense_7/bias%fixed_adjacency_graph_convolution_2/A*fixed_adjacency_graph_convolution_2/kernel(fixed_adjacency_graph_convolution_2/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biasdense_6/kerneldense_6/bias*
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
GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_56415
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp>fixed_adjacency_graph_convolution_2/kernel/Read/ReadVariableOp<fixed_adjacency_graph_convolution_2/bias/Read/ReadVariableOp-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_2/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp9fixed_adjacency_graph_convolution_2/A/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_2/kernel/m/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_2/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpEAdam/fixed_adjacency_graph_convolution_2/kernel/v/Read/ReadVariableOpCAdam/fixed_adjacency_graph_convolution_2/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOpConst*2
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
GPU 2J 8? *'
f"R 
__inference__traced_save_58672
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate*fixed_adjacency_graph_convolution_2/kernel(fixed_adjacency_graph_convolution_2/biaslstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biasdense_6/kerneldense_6/bias%fixed_adjacency_graph_convolution_2/Atotalcounttotal_1count_1Adam/dense_7/kernel/mAdam/dense_7/bias/m1Adam/fixed_adjacency_graph_convolution_2/kernel/m/Adam/fixed_adjacency_graph_convolution_2/bias/m Adam/lstm_2/lstm_cell_2/kernel/m*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mAdam/lstm_2/lstm_cell_2/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/vAdam/dense_7/bias/v1Adam/fixed_adjacency_graph_convolution_2/kernel/v/Adam/fixed_adjacency_graph_convolution_2/bias/v Adam/lstm_2/lstm_cell_2/kernel/v*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vAdam/lstm_2/lstm_cell_2/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/v*1
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_58793??!
?
|
'__inference_dense_6_layer_call_fn_58442

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
GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_559342
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
?
E
)__inference_dropout_6_layer_call_fn_57093

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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561512
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
 
_user_specified_nameinputs
?
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54887

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
?
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58473

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
?
?
&__inference_lstm_2_layer_call_fn_58075

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
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_558632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?b
?
(T-GCN-WX_model_9_lstm_2_while_body_54712L
Ht_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_loop_counterR
Nt_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_maximum_iterations-
)t_gcn_wx_model_9_lstm_2_while_placeholder/
+t_gcn_wx_model_9_lstm_2_while_placeholder_1/
+t_gcn_wx_model_9_lstm_2_while_placeholder_2/
+t_gcn_wx_model_9_lstm_2_while_placeholder_3K
Gt_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_strided_slice_1_0?
?t_gcn_wx_model_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0N
Jt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0P
Lt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0O
Kt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
&t_gcn_wx_model_9_lstm_2_while_identity,
(t_gcn_wx_model_9_lstm_2_while_identity_1,
(t_gcn_wx_model_9_lstm_2_while_identity_2,
(t_gcn_wx_model_9_lstm_2_while_identity_3,
(t_gcn_wx_model_9_lstm_2_while_identity_4,
(t_gcn_wx_model_9_lstm_2_while_identity_5I
Et_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_strided_slice_1?
?t_gcn_wx_model_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensorL
Ht_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceN
Jt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceM
It_gcn_wx_model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource??@T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp??T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?AT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
OT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2Q
OT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
AT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?t_gcn_wx_model_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0)t_gcn_wx_model_9_lstm_2_while_placeholderXT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02C
AT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem?
?T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpJt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02A
?T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?
0T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMulMatMulHT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0GT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul?
AT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpLt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02C
AT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
2T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1MatMul+t_gcn_wx_model_9_lstm_2_while_placeholder_2IT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1?
-T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/addAddV2:T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul:product:0<T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2/
-T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add?
@T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpKt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02B
@T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?
1T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAddBiasAdd1T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add:z:0HT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd?
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :21
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Const?
9T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2;
9T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split/split_dim?
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/splitSplitBT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split/split_dim:output:0:T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split21
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split?
1T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/SigmoidSigmoid8T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????23
1T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid?
3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid8T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????25
3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_1?
-T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mulMul7T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_1:y:0+t_gcn_wx_model_9_lstm_2_while_placeholder_3*
T0*(
_output_shapes
:??????????2/
-T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul?
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_1Mul5T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid:y:08T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????21
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_1?
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add_1AddV21T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul:z:03T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????21
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add_1?
3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid8T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????25
3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_2?
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_2Mul7T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/Sigmoid_2:y:03T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????21
/T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_2?
BT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+t_gcn_wx_model_9_lstm_2_while_placeholder_1)t_gcn_wx_model_9_lstm_2_while_placeholder3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype02D
BT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItem?
#T-GCN-WX/model_9/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#T-GCN-WX/model_9/lstm_2/while/add/y?
!T-GCN-WX/model_9/lstm_2/while/addAddV2)t_gcn_wx_model_9_lstm_2_while_placeholder,T-GCN-WX/model_9/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/model_9/lstm_2/while/add?
%T-GCN-WX/model_9/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%T-GCN-WX/model_9/lstm_2/while/add_1/y?
#T-GCN-WX/model_9/lstm_2/while/add_1AddV2Ht_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_loop_counter.T-GCN-WX/model_9/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_9/lstm_2/while/add_1?
&T-GCN-WX/model_9/lstm_2/while/IdentityIdentity'T-GCN-WX/model_9/lstm_2/while/add_1:z:0A^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&T-GCN-WX/model_9/lstm_2/while/Identity?
(T-GCN-WX/model_9/lstm_2/while/Identity_1IdentityNt_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_maximum_iterationsA^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_9/lstm_2/while/Identity_1?
(T-GCN-WX/model_9/lstm_2/while/Identity_2Identity%T-GCN-WX/model_9/lstm_2/while/add:z:0A^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_9/lstm_2/while/Identity_2?
(T-GCN-WX/model_9/lstm_2/while/Identity_3IdentityRT-GCN-WX/model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(T-GCN-WX/model_9/lstm_2/while/Identity_3?
(T-GCN-WX/model_9/lstm_2/while/Identity_4Identity3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/mul_2:z:0A^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2*
(T-GCN-WX/model_9/lstm_2/while/Identity_4?
(T-GCN-WX/model_9/lstm_2/while/Identity_5Identity3T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/add_1:z:0A^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpB^T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2*
(T-GCN-WX/model_9/lstm_2/while/Identity_5"Y
&t_gcn_wx_model_9_lstm_2_while_identity/T-GCN-WX/model_9/lstm_2/while/Identity:output:0"]
(t_gcn_wx_model_9_lstm_2_while_identity_11T-GCN-WX/model_9/lstm_2/while/Identity_1:output:0"]
(t_gcn_wx_model_9_lstm_2_while_identity_21T-GCN-WX/model_9/lstm_2/while/Identity_2:output:0"]
(t_gcn_wx_model_9_lstm_2_while_identity_31T-GCN-WX/model_9/lstm_2/while/Identity_3:output:0"]
(t_gcn_wx_model_9_lstm_2_while_identity_41T-GCN-WX/model_9/lstm_2/while/Identity_4:output:0"]
(t_gcn_wx_model_9_lstm_2_while_identity_51T-GCN-WX/model_9/lstm_2/while/Identity_5:output:0"?
It_gcn_wx_model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resourceKt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"?
Jt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceLt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Ht_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceJt_gcn_wx_model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"?
Et_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_strided_slice_1Gt_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_strided_slice_1_0"?
?t_gcn_wx_model_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor?t_gcn_wx_model_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_t_gcn_wx_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2?
@T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp@T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2?
?T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?T-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2?
AT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpAT-GCN-WX/model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
C__inference_fixed_adjacency_graph_convolution_2_layer_call_fn_57718
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
GPU 2J 8? *g
fbR`
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_555022
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
?
?
(T-GCN-WX_model_9_lstm_2_while_cond_54711L
Ht_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_loop_counterR
Nt_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_maximum_iterations-
)t_gcn_wx_model_9_lstm_2_while_placeholder/
+t_gcn_wx_model_9_lstm_2_while_placeholder_1/
+t_gcn_wx_model_9_lstm_2_while_placeholder_2/
+t_gcn_wx_model_9_lstm_2_while_placeholder_3N
Jt_gcn_wx_model_9_lstm_2_while_less_t_gcn_wx_model_9_lstm_2_strided_slice_1c
_t_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_cond_54711___redundant_placeholder0c
_t_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_cond_54711___redundant_placeholder1c
_t_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_cond_54711___redundant_placeholder2c
_t_gcn_wx_model_9_lstm_2_while_t_gcn_wx_model_9_lstm_2_while_cond_54711___redundant_placeholder3*
&t_gcn_wx_model_9_lstm_2_while_identity
?
"T-GCN-WX/model_9/lstm_2/while/LessLess)t_gcn_wx_model_9_lstm_2_while_placeholderJt_gcn_wx_model_9_lstm_2_while_less_t_gcn_wx_model_9_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_9/lstm_2/while/Less?
&T-GCN-WX/model_9/lstm_2/while/IdentityIdentity&T-GCN-WX/model_9/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2(
&T-GCN-WX/model_9/lstm_2/while/Identity"Y
&t_gcn_wx_model_9_lstm_2_while_identity/T-GCN-WX/model_9/lstm_2/while/Identity:output:0*U
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
?
?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56273
input_13
dense_7_56248
dense_7_56250
model_9_56255
model_9_56257
model_9_56259
model_9_56261
model_9_56263
model_9_56265
model_9_56267
model_9_56269
identity??dense_7/StatefulPartitionedCall?model_9/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_7_56248dense_7_56250*
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
GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_561182!
dense_7/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561512
dropout_6/PartitionedCall?
reshape_16/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_561772
reshape_16/PartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0model_9_56255model_9_56257model_9_56259model_9_56261model_9_56263model_9_56265model_9_56267model_9_56269*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560652!
model_9/StatefulPartitionedCall?
IdentityIdentity(model_9/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall ^model_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_13
?
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58504

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
?
?
+__inference_lstm_cell_2_layer_call_fn_58538

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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_549182
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
?@
?
while_body_57970
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
?@
?
while_body_58141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
?R
?
__inference__traced_save_58672
file_prefix-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopI
Esavev2_fixed_adjacency_graph_convolution_2_kernel_read_readvariableopG
Csavev2_fixed_adjacency_graph_convolution_2_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableopD
@savev2_fixed_adjacency_graph_convolution_2_a_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_2_kernel_m_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_2_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableopP
Lsavev2_adam_fixed_adjacency_graph_convolution_2_kernel_v_read_readvariableopN
Jsavev2_adam_fixed_adjacency_graph_convolution_2_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopEsavev2_fixed_adjacency_graph_convolution_2_kernel_read_readvariableopCsavev2_fixed_adjacency_graph_convolution_2_bias_read_readvariableop4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop@savev2_fixed_adjacency_graph_convolution_2_a_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_2_kernel_m_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_2_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopLsavev2_adam_fixed_adjacency_graph_convolution_2_kernel_v_read_readvariableopJsavev2_adam_fixed_adjacency_graph_convolution_2_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::: : : : : ::F:	F?:
??:?:	?F:F:FF: : : : ::::F:	F?:
??:?:	?F:F::::F:	F?:
??:?:	?F:F: 2(
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
:	F?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?F: 
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
:	F?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?F: 
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
:	F?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?F: %

_output_shapes
:F:&

_output_shapes
: 
??
?
B__inference_model_9_layer_call_and_return_conditional_losses_57594

inputsG
Cfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_2_add_readvariableop_resource5
1lstm_2_lstm_cell_2_matmul_readvariableop_resource7
3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource6
2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_2/add/ReadVariableOp?>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?lstm_2/while?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinputs(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDimsx
reshape_17/ShapeShape$tf.expand_dims_4/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshape$tf.expand_dims_4/ExpandDims:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_17/Reshape?
2fixed_adjacency_graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_2/transpose/perm?
-fixed_adjacency_graph_convolution_2/transpose	Transposereshape_17/Reshape:output:0;fixed_adjacency_graph_convolution_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/transpose?
)fixed_adjacency_graph_convolution_2/ShapeShape1fixed_adjacency_graph_convolution_2/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_2/Shape?
+fixed_adjacency_graph_convolution_2/unstackUnpack2fixed_adjacency_graph_convolution_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_2/unstack?
:fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_2/Shape_1?
-fixed_adjacency_graph_convolution_2/unstack_1Unpack4fixed_adjacency_graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_1?
1fixed_adjacency_graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_2/Reshape/shape?
+fixed_adjacency_graph_convolution_2/ReshapeReshape1fixed_adjacency_graph_convolution_2/transpose:y:0:fixed_adjacency_graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_2/Reshape?
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_2/transpose_1/perm?
/fixed_adjacency_graph_convolution_2/transpose_1	TransposeFfixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_2/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_2/transpose_1?
3fixed_adjacency_graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_2/Reshape_1/shape?
-fixed_adjacency_graph_convolution_2/Reshape_1Reshape3fixed_adjacency_graph_convolution_2/transpose_1:y:0<fixed_adjacency_graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_2/Reshape_1?
*fixed_adjacency_graph_convolution_2/MatMulMatMul4fixed_adjacency_graph_convolution_2/Reshape:output:06fixed_adjacency_graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_2/MatMul?
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_2/Reshape_2/shapePack4fixed_adjacency_graph_convolution_2/unstack:output:0>fixed_adjacency_graph_convolution_2/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_2/Reshape_2/shape?
-fixed_adjacency_graph_convolution_2/Reshape_2Reshape4fixed_adjacency_graph_convolution_2/MatMul:product:0<fixed_adjacency_graph_convolution_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/Reshape_2?
4fixed_adjacency_graph_convolution_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_2/transpose_2/perm?
/fixed_adjacency_graph_convolution_2/transpose_2	Transpose6fixed_adjacency_graph_convolution_2/Reshape_2:output:0=fixed_adjacency_graph_convolution_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_2/transpose_2?
+fixed_adjacency_graph_convolution_2/Shape_2Shape3fixed_adjacency_graph_convolution_2/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_2/Shape_2?
-fixed_adjacency_graph_convolution_2/unstack_2Unpack4fixed_adjacency_graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_2?
:fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02<
:fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2-
+fixed_adjacency_graph_convolution_2/Shape_3?
-fixed_adjacency_graph_convolution_2/unstack_3Unpack4fixed_adjacency_graph_convolution_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_3?
3fixed_adjacency_graph_convolution_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_2/Reshape_3/shape?
-fixed_adjacency_graph_convolution_2/Reshape_3Reshape3fixed_adjacency_graph_convolution_2/transpose_2:y:0<fixed_adjacency_graph_convolution_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_2/Reshape_3?
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02@
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_2/transpose_3/perm?
/fixed_adjacency_graph_convolution_2/transpose_3	TransposeFfixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_2/transpose_3/perm:output:0*
T0*
_output_shapes

:21
/fixed_adjacency_graph_convolution_2/transpose_3?
3fixed_adjacency_graph_convolution_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_2/Reshape_4/shape?
-fixed_adjacency_graph_convolution_2/Reshape_4Reshape3fixed_adjacency_graph_convolution_2/transpose_3:y:0<fixed_adjacency_graph_convolution_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2/
-fixed_adjacency_graph_convolution_2/Reshape_4?
,fixed_adjacency_graph_convolution_2/MatMul_1MatMul6fixed_adjacency_graph_convolution_2/Reshape_3:output:06fixed_adjacency_graph_convolution_2/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2.
,fixed_adjacency_graph_convolution_2/MatMul_1?
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_2/Reshape_5/shapePack6fixed_adjacency_graph_convolution_2/unstack_2:output:0>fixed_adjacency_graph_convolution_2/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_2/Reshape_5/shape?
-fixed_adjacency_graph_convolution_2/Reshape_5Reshape6fixed_adjacency_graph_convolution_2/MatMul_1:product:0<fixed_adjacency_graph_convolution_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/Reshape_5?
6fixed_adjacency_graph_convolution_2/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_2_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_2/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_2/addAddV26fixed_adjacency_graph_convolution_2/Reshape_5:output:0>fixed_adjacency_graph_convolution_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2)
'fixed_adjacency_graph_convolution_2/add
reshape_18/ShapeShape+fixed_adjacency_graph_convolution_2/add:z:0*
T0*
_output_shapes
:2
reshape_18/Shape?
reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_18/strided_slice/stack?
 reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_1?
 reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_2?
reshape_18/strided_sliceStridedSlicereshape_18/Shape:output:0'reshape_18/strided_slice/stack:output:0)reshape_18/strided_slice/stack_1:output:0)reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_18/strided_slicez
reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_18/Reshape/shape/1?
reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_18/Reshape/shape/2z
reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/3?
reshape_18/Reshape/shapePack!reshape_18/strided_slice:output:0#reshape_18/Reshape/shape/1:output:0#reshape_18/Reshape/shape/2:output:0#reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_18/Reshape/shape?
reshape_18/ReshapeReshape+fixed_adjacency_graph_convolution_2/add:z:0!reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
reshape_18/Reshape?
permute_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_4/transpose/perm?
permute_4/transpose	Transposereshape_18/Reshape:output:0!permute_4/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
permute_4/transposek
reshape_19/ShapeShapepermute_4/transpose:y:0*
T0*
_output_shapes
:2
reshape_19/Shape?
reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_19/strided_slice/stack?
 reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_1?
 reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_2?
reshape_19/strided_sliceStridedSlicereshape_19/Shape:output:0'reshape_19/strided_slice/stack:output:0)reshape_19/strided_slice/stack_1:output:0)reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_19/strided_slice?
reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_19/Reshape/shape/1z
reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_19/Reshape/shape/2?
reshape_19/Reshape/shapePack!reshape_19/strided_slice:output:0#reshape_19/Reshape/shape/1:output:0#reshape_19/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_19/Reshape/shape?
reshape_19/ReshapeReshapepermute_4/transpose:y:0!reshape_19/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_19/Reshapeg
lstm_2/ShapeShapereshape_19/Reshape:output:0*
T0*
_output_shapes
:2
lstm_2/Shape?
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack?
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1?
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/mul/y?
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/Less/y?
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/packed/1?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/mul/y?
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/Less/y?
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/packed/1?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeros_1?
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/perm?
lstm_2/transpose	Transposereshape_19/Reshape:output:0lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1?
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stack?
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1?
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1?
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_2/TensorArrayV2/element_shape?
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2?
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensor?
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stack?
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1?
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_2/strided_slice_2?
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/MatMul?
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/MatMul_1?
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/add?
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/BiasAddv
lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/lstm_cell_2/Const?
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_2/split/split_dim?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_2/lstm_cell_2/split?
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid?
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid_1?
lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul?
lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul_1?
lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/add_1?
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid_2?
lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul_2?
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$lstm_2/TensorArrayV2_1/element_shape?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/time?
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counter?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_2_matmul_readvariableop_resource3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_2_while_body_57503*#
condR
lstm_2_while_cond_57502*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_2/while?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStack?
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_2/strided_slice_3/stack?
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1?
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_2/strided_slice_3?
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtime?
dropout_7/IdentityIdentitylstm_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Identity?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout_7/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_6/BiasAddy
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_6/Sigmoid?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_2/add/ReadVariableOp?^fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_2/add/ReadVariableOp6fixed_adjacency_graph_convolution_2/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_56151

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
?
b
)__inference_dropout_6_layer_call_fn_57088

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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561462
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
?
?
&__inference_lstm_2_layer_call_fn_58384
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_552812
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
a
E__inference_reshape_17_layer_call_and_return_conditional_losses_57649

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
?
?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56357

inputs
dense_7_56332
dense_7_56334
model_9_56339
model_9_56341
model_9_56343
model_9_56345
model_9_56347
model_9_56349
model_9_56351
model_9_56353
identity??dense_7/StatefulPartitionedCall?model_9/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_56332dense_7_56334*
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
GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_561182!
dense_7/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561512
dropout_6/PartitionedCall?
reshape_16/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_561772
reshape_16/PartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0model_9_56339model_9_56341model_9_56343model_9_56345model_9_56347model_9_56349model_9_56351model_9_56353*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560652!
model_9/StatefulPartitionedCall?
IdentityIdentity(model_9/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall ^model_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56304

inputs
dense_7_56279
dense_7_56281
model_9_56286
model_9_56288
model_9_56290
model_9_56292
model_9_56294
model_9_56296
model_9_56298
model_9_56300
identity??dense_7/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?model_9/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_56279dense_7_56281*
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
GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_561182!
dense_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561462#
!dropout_6/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_561772
reshape_16/PartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0model_9_56286model_9_56288model_9_56290model_9_56292model_9_56294model_9_56296model_9_56298model_9_56300*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560142!
model_9/StatefulPartitionedCall?
IdentityIdentity(model_9/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall ^model_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_56415
input_13
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
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
 __inference__wrapped_model_548032
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
input_13
?
?
model_9_lstm_2_while_cond_56604:
6model_9_lstm_2_while_model_9_lstm_2_while_loop_counter@
<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations$
 model_9_lstm_2_while_placeholder&
"model_9_lstm_2_while_placeholder_1&
"model_9_lstm_2_while_placeholder_2&
"model_9_lstm_2_while_placeholder_3<
8model_9_lstm_2_while_less_model_9_lstm_2_strided_slice_1Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56604___redundant_placeholder0Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56604___redundant_placeholder1Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56604___redundant_placeholder2Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56604___redundant_placeholder3!
model_9_lstm_2_while_identity
?
model_9/lstm_2/while/LessLess model_9_lstm_2_while_placeholder8model_9_lstm_2_while_less_model_9_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_2/while/Less?
model_9/lstm_2/while/IdentityIdentitymodel_9/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
model_9/lstm_2/while/Identity"G
model_9_lstm_2_while_identity&model_9/lstm_2/while/Identity:output:0*U
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
lstm_2_while_body_57258*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0?
;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0>
:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor;
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource=
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource<
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource??/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_2/while/TensorArrayV2Read/TensorListGetItem?
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_2/while/lstm_cell_2/MatMul?
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
!lstm_2/while/lstm_cell_2/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_2/while/lstm_cell_2/MatMul_1?
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/MatMul:product:0+lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_2/while/lstm_cell_2/add?
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd lstm_2/while/lstm_cell_2/add:z:07lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_2/while/lstm_cell_2/BiasAdd?
lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_2/while/lstm_cell_2/Const?
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_2/while/lstm_cell_2/split/split_dim?
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:0)lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2 
lstm_2/while/lstm_cell_2/split?
 lstm_2/while/lstm_cell_2/SigmoidSigmoid'lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_2/while/lstm_cell_2/Sigmoid?
"lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_2/while/lstm_cell_2/Sigmoid_1?
lstm_2/while/lstm_cell_2/mulMul&lstm_2/while/lstm_cell_2/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_2/while/lstm_cell_2/mul?
lstm_2/while/lstm_cell_2/mul_1Mul$lstm_2/while/lstm_cell_2/Sigmoid:y:0'lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/mul_1?
lstm_2/while/lstm_cell_2/add_1AddV2 lstm_2/while/lstm_cell_2/mul:z:0"lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/add_1?
"lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_2/while/lstm_cell_2/Sigmoid_2?
lstm_2/while/lstm_cell_2/mul_2Mul&lstm_2/while/lstm_cell_2/Sigmoid_2:y:0"lstm_2/while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/mul_2?
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_2/while/TensorArrayV2Write/TensorListSetItemj
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add/y?
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/addn
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add_1/y?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/add_1?
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations0^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_1?
lstm_2/while/Identity_2Identitylstm_2/while/add:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_2?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_3?
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_2:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_2/while/Identity_4?
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_1:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_2/while/Identity_5"7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2b
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
(__inference_T-GCN-WX_layer_call_fn_56327
input_13
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
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8? *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_563042
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
input_13
??
?	
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56703

inputs-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resourceO
Kmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resourceO
Kmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resourceK
Gmodel_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resource=
9model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource?
;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource>
:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource2
.model_9_dense_6_matmul_readvariableop_resource3
/model_9_dense_6_biasadd_readvariableop_resource
identity??dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?&model_9/dense_6/BiasAdd/ReadVariableOp?%model_9/dense_6/MatMul/ReadVariableOp?>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?model_9/lstm_2/while?
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes?
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_7/Tensordot/freeh
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape?
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2?
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod?
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1?
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack?
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
dense_7/Tensordot/transpose?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_7/Tensordot/Reshape?
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/Tensordot/MatMul?
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2?
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
dense_7/Tensordot?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
dense_7/BiasAddw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_6/dropout/Const?
dropout_6/dropout/MulMuldense_7/BiasAdd:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:?????????F2
dropout_6/dropout/Mulz
dropout_6/dropout/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????F*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????F2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????F2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????F2
dropout_6/dropout/Mul_1o
reshape_16/ShapeShapedropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapedropout_6/dropout/Mul_1:z:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_16/Reshape?
'model_9/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/tf.expand_dims_4/ExpandDims/dim?
#model_9/tf.expand_dims_4/ExpandDims
ExpandDimsreshape_16/Reshape:output:00model_9/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2%
#model_9/tf.expand_dims_4/ExpandDims?
model_9/reshape_17/ShapeShape,model_9/tf.expand_dims_4/ExpandDims:output:0*
T0*
_output_shapes
:2
model_9/reshape_17/Shape?
&model_9/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_17/strided_slice/stack?
(model_9/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_17/strided_slice/stack_1?
(model_9/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_17/strided_slice/stack_2?
 model_9/reshape_17/strided_sliceStridedSlice!model_9/reshape_17/Shape:output:0/model_9/reshape_17/strided_slice/stack:output:01model_9/reshape_17/strided_slice/stack_1:output:01model_9/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_17/strided_slice?
"model_9/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_17/Reshape/shape/1?
"model_9/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/reshape_17/Reshape/shape/2?
 model_9/reshape_17/Reshape/shapePack)model_9/reshape_17/strided_slice:output:0+model_9/reshape_17/Reshape/shape/1:output:0+model_9/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_17/Reshape/shape?
model_9/reshape_17/ReshapeReshape,model_9/tf.expand_dims_4/ExpandDims:output:0)model_9/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_9/reshape_17/Reshape?
:model_9/fixed_adjacency_graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:model_9/fixed_adjacency_graph_convolution_2/transpose/perm?
5model_9/fixed_adjacency_graph_convolution_2/transpose	Transpose#model_9/reshape_17/Reshape:output:0Cmodel_9/fixed_adjacency_graph_convolution_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/transpose?
1model_9/fixed_adjacency_graph_convolution_2/ShapeShape9model_9/fixed_adjacency_graph_convolution_2/transpose:y:0*
T0*
_output_shapes
:23
1model_9/fixed_adjacency_graph_convolution_2/Shape?
3model_9/fixed_adjacency_graph_convolution_2/unstackUnpack:model_9/fixed_adjacency_graph_convolution_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num25
3model_9/fixed_adjacency_graph_convolution_2/unstack?
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02D
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOp?
3model_9/fixed_adjacency_graph_convolution_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   25
3model_9/fixed_adjacency_graph_convolution_2/Shape_1?
5model_9/fixed_adjacency_graph_convolution_2/unstack_1Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_1?
9model_9/fixed_adjacency_graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2;
9model_9/fixed_adjacency_graph_convolution_2/Reshape/shape?
3model_9/fixed_adjacency_graph_convolution_2/ReshapeReshape9model_9/fixed_adjacency_graph_convolution_2/transpose:y:0Bmodel_9/fixed_adjacency_graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F25
3model_9/fixed_adjacency_graph_convolution_2/Reshape?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02H
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?
<model_9/fixed_adjacency_graph_convolution_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_1/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_1	TransposeNmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp:value:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_1/perm:output:0*
T0*
_output_shapes

:FF29
7model_9/fixed_adjacency_graph_convolution_2/transpose_1?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_1Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_1:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_1?
2model_9/fixed_adjacency_graph_convolution_2/MatMulMatMul<model_9/fixed_adjacency_graph_convolution_2/Reshape:output:0>model_9/fixed_adjacency_graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F24
2model_9/fixed_adjacency_graph_convolution_2/MatMul?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shapePack<model_9/fixed_adjacency_graph_convolution_2/unstack:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_2Reshape<model_9/fixed_adjacency_graph_convolution_2/MatMul:product:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_2?
<model_9/fixed_adjacency_graph_convolution_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_2/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_2	Transpose>model_9/fixed_adjacency_graph_convolution_2/Reshape_2:output:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F29
7model_9/fixed_adjacency_graph_convolution_2/transpose_2?
3model_9/fixed_adjacency_graph_convolution_2/Shape_2Shape;model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0*
T0*
_output_shapes
:25
3model_9/fixed_adjacency_graph_convolution_2/Shape_2?
5model_9/fixed_adjacency_graph_convolution_2/unstack_2Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_2?
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02D
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOp?
3model_9/fixed_adjacency_graph_convolution_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      25
3model_9/fixed_adjacency_graph_convolution_2/Shape_3?
5model_9/fixed_adjacency_graph_convolution_2/unstack_3Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_3?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_3Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_3?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02H
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?
<model_9/fixed_adjacency_graph_convolution_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_3/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_3	TransposeNmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp:value:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_3/perm:output:0*
T0*
_output_shapes

:29
7model_9/fixed_adjacency_graph_convolution_2/transpose_3?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_4Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_3:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_4?
4model_9/fixed_adjacency_graph_convolution_2/MatMul_1MatMul>model_9/fixed_adjacency_graph_convolution_2/Reshape_3:output:0>model_9/fixed_adjacency_graph_convolution_2/Reshape_4:output:0*
T0*'
_output_shapes
:?????????26
4model_9/fixed_adjacency_graph_convolution_2/MatMul_1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shapePack>model_9/fixed_adjacency_graph_convolution_2/unstack_2:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_5Reshape>model_9/fixed_adjacency_graph_convolution_2/MatMul_1:product:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_5?
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpReadVariableOpGmodel_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resource*
_output_shapes

:F*
dtype02@
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?
/model_9/fixed_adjacency_graph_convolution_2/addAddV2>model_9/fixed_adjacency_graph_convolution_2/Reshape_5:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F21
/model_9/fixed_adjacency_graph_convolution_2/add?
model_9/reshape_18/ShapeShape3model_9/fixed_adjacency_graph_convolution_2/add:z:0*
T0*
_output_shapes
:2
model_9/reshape_18/Shape?
&model_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_18/strided_slice/stack?
(model_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_18/strided_slice/stack_1?
(model_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_18/strided_slice/stack_2?
 model_9/reshape_18/strided_sliceStridedSlice!model_9/reshape_18/Shape:output:0/model_9/reshape_18/strided_slice/stack:output:01model_9/reshape_18/strided_slice/stack_1:output:01model_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_18/strided_slice?
"model_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_18/Reshape/shape/1?
"model_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_9/reshape_18/Reshape/shape/2?
"model_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/reshape_18/Reshape/shape/3?
 model_9/reshape_18/Reshape/shapePack)model_9/reshape_18/strided_slice:output:0+model_9/reshape_18/Reshape/shape/1:output:0+model_9/reshape_18/Reshape/shape/2:output:0+model_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_18/Reshape/shape?
model_9/reshape_18/ReshapeReshape3model_9/fixed_adjacency_graph_convolution_2/add:z:0)model_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
model_9/reshape_18/Reshape?
 model_9/permute_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 model_9/permute_4/transpose/perm?
model_9/permute_4/transpose	Transpose#model_9/reshape_18/Reshape:output:0)model_9/permute_4/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
model_9/permute_4/transpose?
model_9/reshape_19/ShapeShapemodel_9/permute_4/transpose:y:0*
T0*
_output_shapes
:2
model_9/reshape_19/Shape?
&model_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_19/strided_slice/stack?
(model_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_19/strided_slice/stack_1?
(model_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_19/strided_slice/stack_2?
 model_9/reshape_19/strided_sliceStridedSlice!model_9/reshape_19/Shape:output:0/model_9/reshape_19/strided_slice/stack:output:01model_9/reshape_19/strided_slice/stack_1:output:01model_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_19/strided_slice?
"model_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_9/reshape_19/Reshape/shape/1?
"model_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_19/Reshape/shape/2?
 model_9/reshape_19/Reshape/shapePack)model_9/reshape_19/strided_slice:output:0+model_9/reshape_19/Reshape/shape/1:output:0+model_9/reshape_19/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_19/Reshape/shape?
model_9/reshape_19/ReshapeReshapemodel_9/permute_4/transpose:y:0)model_9/reshape_19/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_9/reshape_19/Reshape
model_9/lstm_2/ShapeShape#model_9/reshape_19/Reshape:output:0*
T0*
_output_shapes
:2
model_9/lstm_2/Shape?
"model_9/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_9/lstm_2/strided_slice/stack?
$model_9/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/lstm_2/strided_slice/stack_1?
$model_9/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/lstm_2/strided_slice/stack_2?
model_9/lstm_2/strided_sliceStridedSlicemodel_9/lstm_2/Shape:output:0+model_9/lstm_2/strided_slice/stack:output:0-model_9/lstm_2/strided_slice/stack_1:output:0-model_9/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_2/strided_slice{
model_9/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/mul/y?
model_9/lstm_2/zeros/mulMul%model_9/lstm_2/strided_slice:output:0#model_9/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros/mul}
model_9/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/Less/y?
model_9/lstm_2/zeros/LessLessmodel_9/lstm_2/zeros/mul:z:0$model_9/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros/Less?
model_9/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/packed/1?
model_9/lstm_2/zeros/packedPack%model_9/lstm_2/strided_slice:output:0&model_9/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_2/zeros/packed}
model_9/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/zeros/Const?
model_9/lstm_2/zerosFill$model_9/lstm_2/zeros/packed:output:0#model_9/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_9/lstm_2/zeros
model_9/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros_1/mul/y?
model_9/lstm_2/zeros_1/mulMul%model_9/lstm_2/strided_slice:output:0%model_9/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros_1/mul?
model_9/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros_1/Less/y?
model_9/lstm_2/zeros_1/LessLessmodel_9/lstm_2/zeros_1/mul:z:0&model_9/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros_1/Less?
model_9/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
model_9/lstm_2/zeros_1/packed/1?
model_9/lstm_2/zeros_1/packedPack%model_9/lstm_2/strided_slice:output:0(model_9/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_2/zeros_1/packed?
model_9/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/zeros_1/Const?
model_9/lstm_2/zeros_1Fill&model_9/lstm_2/zeros_1/packed:output:0%model_9/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_9/lstm_2/zeros_1?
model_9/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_9/lstm_2/transpose/perm?
model_9/lstm_2/transpose	Transpose#model_9/reshape_19/Reshape:output:0&model_9/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
model_9/lstm_2/transpose|
model_9/lstm_2/Shape_1Shapemodel_9/lstm_2/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_2/Shape_1?
$model_9/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/lstm_2/strided_slice_1/stack?
&model_9/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_1/stack_1?
&model_9/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_1/stack_2?
model_9/lstm_2/strided_slice_1StridedSlicemodel_9/lstm_2/Shape_1:output:0-model_9/lstm_2/strided_slice_1/stack:output:0/model_9/lstm_2/strided_slice_1/stack_1:output:0/model_9/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_9/lstm_2/strided_slice_1?
*model_9/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_9/lstm_2/TensorArrayV2/element_shape?
model_9/lstm_2/TensorArrayV2TensorListReserve3model_9/lstm_2/TensorArrayV2/element_shape:output:0'model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_2/TensorArrayV2?
Dmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2F
Dmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
6model_9/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_2/transpose:y:0Mmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor?
$model_9/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/lstm_2/strided_slice_2/stack?
&model_9/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_2/stack_1?
&model_9/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_2/stack_2?
model_9/lstm_2/strided_slice_2StridedSlicemodel_9/lstm_2/transpose:y:0-model_9/lstm_2/strided_slice_2/stack:output:0/model_9/lstm_2/strided_slice_2/stack_1:output:0/model_9/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2 
model_9/lstm_2/strided_slice_2?
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype022
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?
!model_9/lstm_2/lstm_cell_2/MatMulMatMul'model_9/lstm_2/strided_slice_2:output:08model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_9/lstm_2/lstm_cell_2/MatMul?
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?
#model_9/lstm_2/lstm_cell_2/MatMul_1MatMulmodel_9/lstm_2/zeros:output:0:model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_9/lstm_2/lstm_cell_2/MatMul_1?
model_9/lstm_2/lstm_cell_2/addAddV2+model_9/lstm_2/lstm_cell_2/MatMul:product:0-model_9/lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
model_9/lstm_2/lstm_cell_2/add?
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?
"model_9/lstm_2/lstm_cell_2/BiasAddBiasAdd"model_9/lstm_2/lstm_cell_2/add:z:09model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_9/lstm_2/lstm_cell_2/BiasAdd?
 model_9/lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/lstm_2/lstm_cell_2/Const?
*model_9/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_9/lstm_2/lstm_cell_2/split/split_dim?
 model_9/lstm_2/lstm_cell_2/splitSplit3model_9/lstm_2/lstm_cell_2/split/split_dim:output:0+model_9/lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 model_9/lstm_2/lstm_cell_2/split?
"model_9/lstm_2/lstm_cell_2/SigmoidSigmoid)model_9/lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2$
"model_9/lstm_2/lstm_cell_2/Sigmoid?
$model_9/lstm_2/lstm_cell_2/Sigmoid_1Sigmoid)model_9/lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/lstm_cell_2/Sigmoid_1?
model_9/lstm_2/lstm_cell_2/mulMul(model_9/lstm_2/lstm_cell_2/Sigmoid_1:y:0model_9/lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:??????????2 
model_9/lstm_2/lstm_cell_2/mul?
 model_9/lstm_2/lstm_cell_2/mul_1Mul&model_9/lstm_2/lstm_cell_2/Sigmoid:y:0)model_9/lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/mul_1?
 model_9/lstm_2/lstm_cell_2/add_1AddV2"model_9/lstm_2/lstm_cell_2/mul:z:0$model_9/lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/add_1?
$model_9/lstm_2/lstm_cell_2/Sigmoid_2Sigmoid)model_9/lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/lstm_cell_2/Sigmoid_2?
 model_9/lstm_2/lstm_cell_2/mul_2Mul(model_9/lstm_2/lstm_cell_2/Sigmoid_2:y:0$model_9/lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/mul_2?
,model_9/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2.
,model_9/lstm_2/TensorArrayV2_1/element_shape?
model_9/lstm_2/TensorArrayV2_1TensorListReserve5model_9/lstm_2/TensorArrayV2_1/element_shape:output:0'model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_9/lstm_2/TensorArrayV2_1l
model_9/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_2/time?
'model_9/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/lstm_2/while/maximum_iterations?
!model_9/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_9/lstm_2/while/loop_counter?
model_9/lstm_2/whileWhile*model_9/lstm_2/while/loop_counter:output:00model_9/lstm_2/while/maximum_iterations:output:0model_9/lstm_2/time:output:0'model_9/lstm_2/TensorArrayV2_1:handle:0model_9/lstm_2/zeros:output:0model_9/lstm_2/zeros_1:output:0'model_9/lstm_2/strided_slice_1:output:0Fmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_9_lstm_2_while_body_56605*+
cond#R!
model_9_lstm_2_while_cond_56604*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
model_9/lstm_2/while?
?model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape?
1model_9/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_2/while:output:3Hmodel_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype023
1model_9/lstm_2/TensorArrayV2Stack/TensorListStack?
$model_9/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$model_9/lstm_2/strided_slice_3/stack?
&model_9/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/lstm_2/strided_slice_3/stack_1?
&model_9/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_3/stack_2?
model_9/lstm_2/strided_slice_3StridedSlice:model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-model_9/lstm_2/strided_slice_3/stack:output:0/model_9/lstm_2/strided_slice_3/stack_1:output:0/model_9/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2 
model_9/lstm_2/strided_slice_3?
model_9/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_9/lstm_2/transpose_1/perm?
model_9/lstm_2/transpose_1	Transpose:model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(model_9/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
model_9/lstm_2/transpose_1?
model_9/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/runtime?
model_9/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2!
model_9/dropout_7/dropout/Const?
model_9/dropout_7/dropout/MulMul'model_9/lstm_2/strided_slice_3:output:0(model_9/dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
model_9/dropout_7/dropout/Mul?
model_9/dropout_7/dropout/ShapeShape'model_9/lstm_2/strided_slice_3:output:0*
T0*
_output_shapes
:2!
model_9/dropout_7/dropout/Shape?
6model_9/dropout_7/dropout/random_uniform/RandomUniformRandomUniform(model_9/dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype028
6model_9/dropout_7/dropout/random_uniform/RandomUniform?
(model_9/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2*
(model_9/dropout_7/dropout/GreaterEqual/y?
&model_9/dropout_7/dropout/GreaterEqualGreaterEqual?model_9/dropout_7/dropout/random_uniform/RandomUniform:output:01model_9/dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2(
&model_9/dropout_7/dropout/GreaterEqual?
model_9/dropout_7/dropout/CastCast*model_9/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
model_9/dropout_7/dropout/Cast?
model_9/dropout_7/dropout/Mul_1Mul!model_9/dropout_7/dropout/Mul:z:0"model_9/dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2!
model_9/dropout_7/dropout/Mul_1?
%model_9/dense_6/MatMul/ReadVariableOpReadVariableOp.model_9_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02'
%model_9/dense_6/MatMul/ReadVariableOp?
model_9/dense_6/MatMulMatMul#model_9/dropout_7/dropout/Mul_1:z:0-model_9/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/MatMul?
&model_9/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_9_dense_6_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_9/dense_6/BiasAdd/ReadVariableOp?
model_9/dense_6/BiasAddBiasAdd model_9/dense_6/MatMul:product:0.model_9/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/BiasAdd?
model_9/dense_6/SigmoidSigmoid model_9/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/Sigmoid?
IdentityIdentitymodel_9/dense_6/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp'^model_9/dense_6/BiasAdd/ReadVariableOp&^model_9/dense_6/MatMul/ReadVariableOp?^model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpG^model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpG^model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2^model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp1^model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp3^model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^model_9/lstm_2/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2P
&model_9/dense_6/BiasAdd/ReadVariableOp&model_9/dense_6/BiasAdd/ReadVariableOp2N
%model_9/dense_6/MatMul/ReadVariableOp%model_9/dense_6/MatMul/ReadVariableOp2?
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp2?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpFmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp2?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpFmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2f
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2d
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp2h
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2,
model_9/lstm_2/whilemodel_9/lstm_2/while:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
&__inference_lstm_2_layer_call_fn_58064

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
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_557142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_56146

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
?&
?
B__inference_model_9_layer_call_and_return_conditional_losses_55981
input_14-
)fixed_adjacency_graph_convolution_2_55957-
)fixed_adjacency_graph_convolution_2_55959-
)fixed_adjacency_graph_convolution_2_55961
lstm_2_55967
lstm_2_55969
lstm_2_55971
dense_6_55975
dense_6_55977
identity??dense_6/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinput_14(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDims?
reshape_17/PartitionedCallPartitionedCall$tf.expand_dims_4/ExpandDims:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_554412
reshape_17/PartitionedCall?
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_17/PartitionedCall:output:0)fixed_adjacency_graph_convolution_2_55957)fixed_adjacency_graph_convolution_2_55959)fixed_adjacency_graph_convolution_2_55961*
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
GPU 2J 8? *g
fbR`
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_555022=
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_18_layer_call_and_return_conditional_losses_555362
reshape_18/PartitionedCall?
permute_4/PartitionedCallPartitionedCall#reshape_18/PartitionedCall:output:0*
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
D__inference_permute_4_layer_call_and_return_conditional_losses_548102
permute_4/PartitionedCall?
reshape_19/PartitionedCallPartitionedCall"permute_4/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_19_layer_call_and_return_conditional_losses_555582
reshape_19/PartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_19/PartitionedCall:output:0lstm_2_55967lstm_2_55969lstm_2_55971*
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_558632 
lstm_2/StatefulPartitionedCall?
dropout_7/PartitionedCallPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559102
dropout_7/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_6_55975dense_6_55977*
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
GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_559342!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_2/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_14
ݟ
?
!__inference__traced_restore_58793
file_prefix#
assignvariableop_dense_7_kernel#
assignvariableop_1_dense_7_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rateA
=assignvariableop_7_fixed_adjacency_graph_convolution_2_kernel?
;assignvariableop_8_fixed_adjacency_graph_convolution_2_bias0
,assignvariableop_9_lstm_2_lstm_cell_2_kernel;
7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernel/
+assignvariableop_11_lstm_2_lstm_cell_2_bias&
"assignvariableop_12_dense_6_kernel$
 assignvariableop_13_dense_6_bias=
9assignvariableop_14_fixed_adjacency_graph_convolution_2_a
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1-
)assignvariableop_19_adam_dense_7_kernel_m+
'assignvariableop_20_adam_dense_7_bias_mI
Eassignvariableop_21_adam_fixed_adjacency_graph_convolution_2_kernel_mG
Cassignvariableop_22_adam_fixed_adjacency_graph_convolution_2_bias_m8
4assignvariableop_23_adam_lstm_2_lstm_cell_2_kernel_mB
>assignvariableop_24_adam_lstm_2_lstm_cell_2_recurrent_kernel_m6
2assignvariableop_25_adam_lstm_2_lstm_cell_2_bias_m-
)assignvariableop_26_adam_dense_6_kernel_m+
'assignvariableop_27_adam_dense_6_bias_m-
)assignvariableop_28_adam_dense_7_kernel_v+
'assignvariableop_29_adam_dense_7_bias_vI
Eassignvariableop_30_adam_fixed_adjacency_graph_convolution_2_kernel_vG
Cassignvariableop_31_adam_fixed_adjacency_graph_convolution_2_bias_v8
4assignvariableop_32_adam_lstm_2_lstm_cell_2_kernel_vB
>assignvariableop_33_adam_lstm_2_lstm_cell_2_recurrent_kernel_v6
2assignvariableop_34_adam_lstm_2_lstm_cell_2_bias_v-
)assignvariableop_35_adam_dense_6_kernel_v+
'assignvariableop_36_adam_dense_6_bias_v
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
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp=assignvariableop_7_fixed_adjacency_graph_convolution_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp;assignvariableop_8_fixed_adjacency_graph_convolution_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_2_lstm_cell_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_2_lstm_cell_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_fixed_adjacency_graph_convolution_2_aIdentity_14:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_7_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_7_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpEassignvariableop_21_adam_fixed_adjacency_graph_convolution_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpCassignvariableop_22_adam_fixed_adjacency_graph_convolution_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_2_lstm_cell_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_2_lstm_cell_2_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_2_lstm_cell_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_6_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_6_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_7_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_7_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpEassignvariableop_30_adam_fixed_adjacency_graph_convolution_2_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpCassignvariableop_31_adam_fixed_adjacency_graph_convolution_2_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_2_lstm_cell_2_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_lstm_2_lstm_cell_2_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_lstm_2_lstm_cell_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_6_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_6_bias_vIdentity_36:output:0"/device:CPU:0*
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
?
F
*__inference_reshape_16_layer_call_fn_57111

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
GPU 2J 8? *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_561772
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
?D
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_55413

inputs
lstm_cell_2_55331
lstm_cell_2_55333
lstm_cell_2_55335
identity??#lstm_cell_2/StatefulPartitionedCall?whileD
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
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_55331lstm_cell_2_55333lstm_cell_2_55335*
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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_549182%
#lstm_cell_2/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_55331lstm_cell_2_55333lstm_cell_2_55335*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55344*
condR
while_cond_55343*M
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?X
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_55714

inputs.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55631*
condR
while_cond_55630*M
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
:??????????*
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
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
a
E__inference_reshape_18_layer_call_and_return_conditional_losses_57732

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
?'
?
B__inference_model_9_layer_call_and_return_conditional_losses_55951
input_14-
)fixed_adjacency_graph_convolution_2_55515-
)fixed_adjacency_graph_convolution_2_55517-
)fixed_adjacency_graph_convolution_2_55519
lstm_2_55886
lstm_2_55888
lstm_2_55890
dense_6_55945
dense_6_55947
identity??dense_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinput_14(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDims?
reshape_17/PartitionedCallPartitionedCall$tf.expand_dims_4/ExpandDims:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_554412
reshape_17/PartitionedCall?
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_17/PartitionedCall:output:0)fixed_adjacency_graph_convolution_2_55515)fixed_adjacency_graph_convolution_2_55517)fixed_adjacency_graph_convolution_2_55519*
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
GPU 2J 8? *g
fbR`
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_555022=
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_18_layer_call_and_return_conditional_losses_555362
reshape_18/PartitionedCall?
permute_4/PartitionedCallPartitionedCall#reshape_18/PartitionedCall:output:0*
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
D__inference_permute_4_layer_call_and_return_conditional_losses_548102
permute_4/PartitionedCall?
reshape_19/PartitionedCallPartitionedCall"permute_4/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_19_layer_call_and_return_conditional_losses_555582
reshape_19/PartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_19/PartitionedCall:output:0lstm_2_55886lstm_2_55888lstm_2_55890*
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_557142 
lstm_2/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559052#
!dropout_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_6_55945dense_6_55947*
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
GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_559342!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_2/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????F
"
_user_specified_name
input_14
?
?
(__inference_T-GCN-WX_layer_call_fn_57002

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
GPU 2J 8? *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_563042
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
?
?
'__inference_model_9_layer_call_fn_56033
input_14
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560142
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
input_14
?
?
while_cond_55630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55630___redundant_placeholder03
/while_while_cond_55630___redundant_placeholder13
/while_while_cond_55630___redundant_placeholder23
/while_while_cond_55630___redundant_placeholder3
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
?
+__inference_lstm_cell_2_layer_call_fn_58521

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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_548872
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
?
E
)__inference_permute_4_layer_call_fn_54816

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
D__inference_permute_4_layer_call_and_return_conditional_losses_548102
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
?U
?
model_9_lstm_2_while_body_56605:
6model_9_lstm_2_while_model_9_lstm_2_while_loop_counter@
<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations$
 model_9_lstm_2_while_placeholder&
"model_9_lstm_2_while_placeholder_1&
"model_9_lstm_2_while_placeholder_2&
"model_9_lstm_2_while_placeholder_39
5model_9_lstm_2_while_model_9_lstm_2_strided_slice_1_0u
qmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0E
Amodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0G
Cmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0F
Bmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0!
model_9_lstm_2_while_identity#
model_9_lstm_2_while_identity_1#
model_9_lstm_2_while_identity_2#
model_9_lstm_2_while_identity_3#
model_9_lstm_2_while_identity_4#
model_9_lstm_2_while_identity_57
3model_9_lstm_2_while_model_9_lstm_2_strided_slice_1s
omodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensorC
?model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceE
Amodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceD
@model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource??7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
Fmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2H
Fmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 model_9_lstm_2_while_placeholderOmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02:
8model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem?
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpAmodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype028
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?
'model_9/lstm_2/while/lstm_cell_2/MatMulMatMul?model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'model_9/lstm_2/while/lstm_cell_2/MatMul?
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpCmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02:
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
)model_9/lstm_2/while/lstm_cell_2/MatMul_1MatMul"model_9_lstm_2_while_placeholder_2@model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_9/lstm_2/while/lstm_cell_2/MatMul_1?
$model_9/lstm_2/while/lstm_cell_2/addAddV21model_9/lstm_2/while/lstm_cell_2/MatMul:product:03model_9/lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/while/lstm_cell_2/add?
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpBmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype029
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?
(model_9/lstm_2/while/lstm_cell_2/BiasAddBiasAdd(model_9/lstm_2/while/lstm_cell_2/add:z:0?model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_9/lstm_2/while/lstm_cell_2/BiasAdd?
&model_9/lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_9/lstm_2/while/lstm_cell_2/Const?
0model_9/lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_9/lstm_2/while/lstm_cell_2/split/split_dim?
&model_9/lstm_2/while/lstm_cell_2/splitSplit9model_9/lstm_2/while/lstm_cell_2/split/split_dim:output:01model_9/lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2(
&model_9/lstm_2/while/lstm_cell_2/split?
(model_9/lstm_2/while/lstm_cell_2/SigmoidSigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2*
(model_9/lstm_2/while/lstm_cell_2/Sigmoid?
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2,
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_1?
$model_9/lstm_2/while/lstm_cell_2/mulMul.model_9/lstm_2/while/lstm_cell_2/Sigmoid_1:y:0"model_9_lstm_2_while_placeholder_3*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/while/lstm_cell_2/mul?
&model_9/lstm_2/while/lstm_cell_2/mul_1Mul,model_9/lstm_2/while/lstm_cell_2/Sigmoid:y:0/model_9/lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/mul_1?
&model_9/lstm_2/while/lstm_cell_2/add_1AddV2(model_9/lstm_2/while/lstm_cell_2/mul:z:0*model_9/lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/add_1?
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2,
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_2?
&model_9/lstm_2/while/lstm_cell_2/mul_2Mul.model_9/lstm_2/while/lstm_cell_2/Sigmoid_2:y:0*model_9/lstm_2/while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/mul_2?
9model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_9_lstm_2_while_placeholder_1 model_9_lstm_2_while_placeholder*model_9/lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItemz
model_9/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_2/while/add/y?
model_9/lstm_2/while/addAddV2 model_9_lstm_2_while_placeholder#model_9/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/while/add~
model_9/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_2/while/add_1/y?
model_9/lstm_2/while/add_1AddV26model_9_lstm_2_while_model_9_lstm_2_while_loop_counter%model_9/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/while/add_1?
model_9/lstm_2/while/IdentityIdentitymodel_9/lstm_2/while/add_1:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_9/lstm_2/while/Identity?
model_9/lstm_2/while/Identity_1Identity<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations8^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_1?
model_9/lstm_2/while/Identity_2Identitymodel_9/lstm_2/while/add:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_2?
model_9/lstm_2/while/Identity_3IdentityImodel_9/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_3?
model_9/lstm_2/while/Identity_4Identity*model_9/lstm_2/while/lstm_cell_2/mul_2:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_9/lstm_2/while/Identity_4?
model_9/lstm_2/while/Identity_5Identity*model_9/lstm_2/while/lstm_cell_2/add_1:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_9/lstm_2/while/Identity_5"G
model_9_lstm_2_while_identity&model_9/lstm_2/while/Identity:output:0"K
model_9_lstm_2_while_identity_1(model_9/lstm_2/while/Identity_1:output:0"K
model_9_lstm_2_while_identity_2(model_9/lstm_2/while/Identity_2:output:0"K
model_9_lstm_2_while_identity_3(model_9/lstm_2/while/Identity_3:output:0"K
model_9_lstm_2_while_identity_4(model_9/lstm_2/while/Identity_4:output:0"K
model_9_lstm_2_while_identity_5(model_9/lstm_2/while/Identity_5:output:0"?
@model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resourceBmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"?
Amodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceCmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"?
?model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceAmodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"l
3model_9_lstm_2_while_model_9_lstm_2_strided_slice_15model_9_lstm_2_while_model_9_lstm_2_strided_slice_1_0"?
omodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensorqmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2r
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2p
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2t
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
?
lstm_2_while_cond_57257*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_57257___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_57257___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_57257___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_57257___redundant_placeholder3
lstm_2_while_identity
?
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
lstm_2/while/Lessr
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_2/while/Identity"7
lstm_2_while_identitylstm_2/while/Identity:output:0*U
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
?
model_9_lstm_2_while_cond_56885:
6model_9_lstm_2_while_model_9_lstm_2_while_loop_counter@
<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations$
 model_9_lstm_2_while_placeholder&
"model_9_lstm_2_while_placeholder_1&
"model_9_lstm_2_while_placeholder_2&
"model_9_lstm_2_while_placeholder_3<
8model_9_lstm_2_while_less_model_9_lstm_2_strided_slice_1Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56885___redundant_placeholder0Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56885___redundant_placeholder1Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56885___redundant_placeholder2Q
Mmodel_9_lstm_2_while_model_9_lstm_2_while_cond_56885___redundant_placeholder3!
model_9_lstm_2_while_identity
?
model_9/lstm_2/while/LessLess model_9_lstm_2_while_placeholder8model_9_lstm_2_while_less_model_9_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_2/while/Less?
model_9/lstm_2/while/IdentityIdentitymodel_9/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
model_9/lstm_2/while/Identity"G
model_9_lstm_2_while_identity&model_9/lstm_2/while/Identity:output:0*U
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
??
?
B__inference_model_9_layer_call_and_return_conditional_losses_57356

inputsG
Cfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resourceG
Cfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resourceC
?fixed_adjacency_graph_convolution_2_add_readvariableop_resource5
1lstm_2_lstm_cell_2_matmul_readvariableop_resource7
3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource6
2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?6fixed_adjacency_graph_convolution_2/add/ReadVariableOp?>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?lstm_2/while?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinputs(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDimsx
reshape_17/ShapeShape$tf.expand_dims_4/ExpandDims:output:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshape$tf.expand_dims_4/ExpandDims:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_17/Reshape?
2fixed_adjacency_graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2fixed_adjacency_graph_convolution_2/transpose/perm?
-fixed_adjacency_graph_convolution_2/transpose	Transposereshape_17/Reshape:output:0;fixed_adjacency_graph_convolution_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/transpose?
)fixed_adjacency_graph_convolution_2/ShapeShape1fixed_adjacency_graph_convolution_2/transpose:y:0*
T0*
_output_shapes
:2+
)fixed_adjacency_graph_convolution_2/Shape?
+fixed_adjacency_graph_convolution_2/unstackUnpack2fixed_adjacency_graph_convolution_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2-
+fixed_adjacency_graph_convolution_2/unstack?
:fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02<
:fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOp?
+fixed_adjacency_graph_convolution_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2-
+fixed_adjacency_graph_convolution_2/Shape_1?
-fixed_adjacency_graph_convolution_2/unstack_1Unpack4fixed_adjacency_graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_1?
1fixed_adjacency_graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   23
1fixed_adjacency_graph_convolution_2/Reshape/shape?
+fixed_adjacency_graph_convolution_2/ReshapeReshape1fixed_adjacency_graph_convolution_2/transpose:y:0:fixed_adjacency_graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2-
+fixed_adjacency_graph_convolution_2/Reshape?
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02@
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?
4fixed_adjacency_graph_convolution_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_2/transpose_1/perm?
/fixed_adjacency_graph_convolution_2/transpose_1	TransposeFfixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_2/transpose_1/perm:output:0*
T0*
_output_shapes

:FF21
/fixed_adjacency_graph_convolution_2/transpose_1?
3fixed_adjacency_graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????25
3fixed_adjacency_graph_convolution_2/Reshape_1/shape?
-fixed_adjacency_graph_convolution_2/Reshape_1Reshape3fixed_adjacency_graph_convolution_2/transpose_1:y:0<fixed_adjacency_graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2/
-fixed_adjacency_graph_convolution_2/Reshape_1?
*fixed_adjacency_graph_convolution_2/MatMulMatMul4fixed_adjacency_graph_convolution_2/Reshape:output:06fixed_adjacency_graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2,
*fixed_adjacency_graph_convolution_2/MatMul?
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/1?
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_2/Reshape_2/shape/2?
3fixed_adjacency_graph_convolution_2/Reshape_2/shapePack4fixed_adjacency_graph_convolution_2/unstack:output:0>fixed_adjacency_graph_convolution_2/Reshape_2/shape/1:output:0>fixed_adjacency_graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_2/Reshape_2/shape?
-fixed_adjacency_graph_convolution_2/Reshape_2Reshape4fixed_adjacency_graph_convolution_2/MatMul:product:0<fixed_adjacency_graph_convolution_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/Reshape_2?
4fixed_adjacency_graph_convolution_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          26
4fixed_adjacency_graph_convolution_2/transpose_2/perm?
/fixed_adjacency_graph_convolution_2/transpose_2	Transpose6fixed_adjacency_graph_convolution_2/Reshape_2:output:0=fixed_adjacency_graph_convolution_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F21
/fixed_adjacency_graph_convolution_2/transpose_2?
+fixed_adjacency_graph_convolution_2/Shape_2Shape3fixed_adjacency_graph_convolution_2/transpose_2:y:0*
T0*
_output_shapes
:2-
+fixed_adjacency_graph_convolution_2/Shape_2?
-fixed_adjacency_graph_convolution_2/unstack_2Unpack4fixed_adjacency_graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_2?
:fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02<
:fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOp?
+fixed_adjacency_graph_convolution_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2-
+fixed_adjacency_graph_convolution_2/Shape_3?
-fixed_adjacency_graph_convolution_2/unstack_3Unpack4fixed_adjacency_graph_convolution_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2/
-fixed_adjacency_graph_convolution_2/unstack_3?
3fixed_adjacency_graph_convolution_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   25
3fixed_adjacency_graph_convolution_2/Reshape_3/shape?
-fixed_adjacency_graph_convolution_2/Reshape_3Reshape3fixed_adjacency_graph_convolution_2/transpose_2:y:0<fixed_adjacency_graph_convolution_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2/
-fixed_adjacency_graph_convolution_2/Reshape_3?
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpReadVariableOpCfixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02@
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?
4fixed_adjacency_graph_convolution_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       26
4fixed_adjacency_graph_convolution_2/transpose_3/perm?
/fixed_adjacency_graph_convolution_2/transpose_3	TransposeFfixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp:value:0=fixed_adjacency_graph_convolution_2/transpose_3/perm:output:0*
T0*
_output_shapes

:21
/fixed_adjacency_graph_convolution_2/transpose_3?
3fixed_adjacency_graph_convolution_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????25
3fixed_adjacency_graph_convolution_2/Reshape_4/shape?
-fixed_adjacency_graph_convolution_2/Reshape_4Reshape3fixed_adjacency_graph_convolution_2/transpose_3:y:0<fixed_adjacency_graph_convolution_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2/
-fixed_adjacency_graph_convolution_2/Reshape_4?
,fixed_adjacency_graph_convolution_2/MatMul_1MatMul6fixed_adjacency_graph_convolution_2/Reshape_3:output:06fixed_adjacency_graph_convolution_2/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2.
,fixed_adjacency_graph_convolution_2/MatMul_1?
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F27
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/1?
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5fixed_adjacency_graph_convolution_2/Reshape_5/shape/2?
3fixed_adjacency_graph_convolution_2/Reshape_5/shapePack6fixed_adjacency_graph_convolution_2/unstack_2:output:0>fixed_adjacency_graph_convolution_2/Reshape_5/shape/1:output:0>fixed_adjacency_graph_convolution_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:25
3fixed_adjacency_graph_convolution_2/Reshape_5/shape?
-fixed_adjacency_graph_convolution_2/Reshape_5Reshape6fixed_adjacency_graph_convolution_2/MatMul_1:product:0<fixed_adjacency_graph_convolution_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2/
-fixed_adjacency_graph_convolution_2/Reshape_5?
6fixed_adjacency_graph_convolution_2/add/ReadVariableOpReadVariableOp?fixed_adjacency_graph_convolution_2_add_readvariableop_resource*
_output_shapes

:F*
dtype028
6fixed_adjacency_graph_convolution_2/add/ReadVariableOp?
'fixed_adjacency_graph_convolution_2/addAddV26fixed_adjacency_graph_convolution_2/Reshape_5:output:0>fixed_adjacency_graph_convolution_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2)
'fixed_adjacency_graph_convolution_2/add
reshape_18/ShapeShape+fixed_adjacency_graph_convolution_2/add:z:0*
T0*
_output_shapes
:2
reshape_18/Shape?
reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_18/strided_slice/stack?
 reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_1?
 reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_2?
reshape_18/strided_sliceStridedSlicereshape_18/Shape:output:0'reshape_18/strided_slice/stack:output:0)reshape_18/strided_slice/stack_1:output:0)reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_18/strided_slicez
reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_18/Reshape/shape/1?
reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_18/Reshape/shape/2z
reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/3?
reshape_18/Reshape/shapePack!reshape_18/strided_slice:output:0#reshape_18/Reshape/shape/1:output:0#reshape_18/Reshape/shape/2:output:0#reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_18/Reshape/shape?
reshape_18/ReshapeReshape+fixed_adjacency_graph_convolution_2/add:z:0!reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
reshape_18/Reshape?
permute_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
permute_4/transpose/perm?
permute_4/transpose	Transposereshape_18/Reshape:output:0!permute_4/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
permute_4/transposek
reshape_19/ShapeShapepermute_4/transpose:y:0*
T0*
_output_shapes
:2
reshape_19/Shape?
reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_19/strided_slice/stack?
 reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_1?
 reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_2?
reshape_19/strided_sliceStridedSlicereshape_19/Shape:output:0'reshape_19/strided_slice/stack:output:0)reshape_19/strided_slice/stack_1:output:0)reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_19/strided_slice?
reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_19/Reshape/shape/1z
reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_19/Reshape/shape/2?
reshape_19/Reshape/shapePack!reshape_19/strided_slice:output:0#reshape_19/Reshape/shape/1:output:0#reshape_19/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_19/Reshape/shape?
reshape_19/ReshapeReshapepermute_4/transpose:y:0!reshape_19/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_19/Reshapeg
lstm_2/ShapeShapereshape_19/Reshape:output:0*
T0*
_output_shapes
:2
lstm_2/Shape?
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack?
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1?
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/mul/y?
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/Less/y?
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/packed/1?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/mul/y?
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/Less/y?
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/packed/1?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeros_1?
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/perm?
lstm_2/transpose	Transposereshape_19/Reshape:output:0lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1?
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stack?
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1?
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1?
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_2/TensorArrayV2/element_shape?
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2?
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensor?
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stack?
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1?
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2
lstm_2/strided_slice_2?
(lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02*
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/MatMul?
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/MatMul_1?
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/MatMul:product:0%lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/add?
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_2/lstm_cell_2/BiasAddBiasAddlstm_2/lstm_cell_2/add:z:01lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/BiasAddv
lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/lstm_cell_2/Const?
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_2/split/split_dim?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0#lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_2/lstm_cell_2/split?
lstm_2/lstm_cell_2/SigmoidSigmoid!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid?
lstm_2/lstm_cell_2/Sigmoid_1Sigmoid!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid_1?
lstm_2/lstm_cell_2/mulMul lstm_2/lstm_cell_2/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul?
lstm_2/lstm_cell_2/mul_1Mullstm_2/lstm_cell_2/Sigmoid:y:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul_1?
lstm_2/lstm_cell_2/add_1AddV2lstm_2/lstm_cell_2/mul:z:0lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/add_1?
lstm_2/lstm_cell_2/Sigmoid_2Sigmoid!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/Sigmoid_2?
lstm_2/lstm_cell_2/mul_2Mul lstm_2/lstm_cell_2/Sigmoid_2:y:0lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_2/lstm_cell_2/mul_2?
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$lstm_2/TensorArrayV2_1/element_shape?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/time?
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counter?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_2_matmul_readvariableop_resource3lstm_2_lstm_cell_2_matmul_1_readvariableop_resource2lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_2_while_body_57258*#
condR
lstm_2_while_cond_57257*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_2/while?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStack?
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_2/strided_slice_3/stack?
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1?
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_2/strided_slice_3?
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtimew
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_7/dropout/Const?
dropout_7/dropout/MulMullstm_2/strided_slice_3:output:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShapelstm_2/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/dropout/Mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout_7/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_6/BiasAddy
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
dense_6/Sigmoid?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp7^fixed_adjacency_graph_convolution_2/add/ReadVariableOp?^fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?^fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp*^lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_2/MatMul/ReadVariableOp+^lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^lstm_2/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2p
6fixed_adjacency_graph_convolution_2/add/ReadVariableOp6fixed_adjacency_graph_convolution_2/add/ReadVariableOp2?
>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp>fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp2?
>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp>fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2V
)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_2/MatMul/ReadVariableOp(lstm_2/lstm_cell_2/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_57078

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
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_55905

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
?
?
while_cond_55779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55779___redundant_placeholder03
/while_while_cond_55779___redundant_placeholder13
/while_while_cond_55779___redundant_placeholder23
/while_while_cond_55779___redundant_placeholder3
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
lstm_2_while_cond_57502*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_57502___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_57502___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_57502___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_57502___redundant_placeholder3
lstm_2_while_identity
?
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
lstm_2/while/Lessr
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_2/while/Identity"7
lstm_2_while_identitylstm_2/while/Identity:output:0*U
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
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_57083

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
?,
?
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_55502
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
?U
?
model_9_lstm_2_while_body_56886:
6model_9_lstm_2_while_model_9_lstm_2_while_loop_counter@
<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations$
 model_9_lstm_2_while_placeholder&
"model_9_lstm_2_while_placeholder_1&
"model_9_lstm_2_while_placeholder_2&
"model_9_lstm_2_while_placeholder_39
5model_9_lstm_2_while_model_9_lstm_2_strided_slice_1_0u
qmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0E
Amodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0G
Cmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0F
Bmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0!
model_9_lstm_2_while_identity#
model_9_lstm_2_while_identity_1#
model_9_lstm_2_while_identity_2#
model_9_lstm_2_while_identity_3#
model_9_lstm_2_while_identity_4#
model_9_lstm_2_while_identity_57
3model_9_lstm_2_while_model_9_lstm_2_strided_slice_1s
omodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensorC
?model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceE
Amodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceD
@model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource??7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
Fmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2H
Fmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0 model_9_lstm_2_while_placeholderOmodel_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype02:
8model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem?
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpAmodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype028
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?
'model_9/lstm_2/while/lstm_cell_2/MatMulMatMul?model_9/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0>model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'model_9/lstm_2/while/lstm_cell_2/MatMul?
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpCmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02:
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
)model_9/lstm_2/while/lstm_cell_2/MatMul_1MatMul"model_9_lstm_2_while_placeholder_2@model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_9/lstm_2/while/lstm_cell_2/MatMul_1?
$model_9/lstm_2/while/lstm_cell_2/addAddV21model_9/lstm_2/while/lstm_cell_2/MatMul:product:03model_9/lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/while/lstm_cell_2/add?
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpBmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype029
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?
(model_9/lstm_2/while/lstm_cell_2/BiasAddBiasAdd(model_9/lstm_2/while/lstm_cell_2/add:z:0?model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_9/lstm_2/while/lstm_cell_2/BiasAdd?
&model_9/lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_9/lstm_2/while/lstm_cell_2/Const?
0model_9/lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_9/lstm_2/while/lstm_cell_2/split/split_dim?
&model_9/lstm_2/while/lstm_cell_2/splitSplit9model_9/lstm_2/while/lstm_cell_2/split/split_dim:output:01model_9/lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2(
&model_9/lstm_2/while/lstm_cell_2/split?
(model_9/lstm_2/while/lstm_cell_2/SigmoidSigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2*
(model_9/lstm_2/while/lstm_cell_2/Sigmoid?
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2,
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_1?
$model_9/lstm_2/while/lstm_cell_2/mulMul.model_9/lstm_2/while/lstm_cell_2/Sigmoid_1:y:0"model_9_lstm_2_while_placeholder_3*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/while/lstm_cell_2/mul?
&model_9/lstm_2/while/lstm_cell_2/mul_1Mul,model_9/lstm_2/while/lstm_cell_2/Sigmoid:y:0/model_9/lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/mul_1?
&model_9/lstm_2/while/lstm_cell_2/add_1AddV2(model_9/lstm_2/while/lstm_cell_2/mul:z:0*model_9/lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/add_1?
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid/model_9/lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2,
*model_9/lstm_2/while/lstm_cell_2/Sigmoid_2?
&model_9/lstm_2/while/lstm_cell_2/mul_2Mul.model_9/lstm_2/while/lstm_cell_2/Sigmoid_2:y:0*model_9/lstm_2/while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2(
&model_9/lstm_2/while/lstm_cell_2/mul_2?
9model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_9_lstm_2_while_placeholder_1 model_9_lstm_2_while_placeholder*model_9/lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype02;
9model_9/lstm_2/while/TensorArrayV2Write/TensorListSetItemz
model_9/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_2/while/add/y?
model_9/lstm_2/while/addAddV2 model_9_lstm_2_while_placeholder#model_9/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/while/add~
model_9/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_2/while/add_1/y?
model_9/lstm_2/while/add_1AddV26model_9_lstm_2_while_model_9_lstm_2_while_loop_counter%model_9/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/while/add_1?
model_9/lstm_2/while/IdentityIdentitymodel_9/lstm_2/while/add_1:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_9/lstm_2/while/Identity?
model_9/lstm_2/while/Identity_1Identity<model_9_lstm_2_while_model_9_lstm_2_while_maximum_iterations8^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_1?
model_9/lstm_2/while/Identity_2Identitymodel_9/lstm_2/while/add:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_2?
model_9/lstm_2/while/Identity_3IdentityImodel_9/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_9/lstm_2/while/Identity_3?
model_9/lstm_2/while/Identity_4Identity*model_9/lstm_2/while/lstm_cell_2/mul_2:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_9/lstm_2/while/Identity_4?
model_9/lstm_2/while/Identity_5Identity*model_9/lstm_2/while/lstm_cell_2/add_1:z:08^model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7^model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp9^model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2!
model_9/lstm_2/while/Identity_5"G
model_9_lstm_2_while_identity&model_9/lstm_2/while/Identity:output:0"K
model_9_lstm_2_while_identity_1(model_9/lstm_2/while/Identity_1:output:0"K
model_9_lstm_2_while_identity_2(model_9/lstm_2/while/Identity_2:output:0"K
model_9_lstm_2_while_identity_3(model_9/lstm_2/while/Identity_3:output:0"K
model_9_lstm_2_while_identity_4(model_9/lstm_2/while/Identity_4:output:0"K
model_9_lstm_2_while_identity_5(model_9/lstm_2/while/Identity_5:output:0"?
@model_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resourceBmodel_9_lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"?
Amodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resourceCmodel_9_lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"?
?model_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resourceAmodel_9_lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"l
3model_9_lstm_2_while_model_9_lstm_2_strided_slice_15model_9_lstm_2_while_model_9_lstm_2_strided_slice_1_0"?
omodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensorqmodel_9_lstm_2_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2r
7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp7model_9/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2p
6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp6model_9/lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2t
8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp8model_9/lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
??
?
 __inference__wrapped_model_54803
input_136
2t_gcn_wx_dense_7_tensordot_readvariableop_resource4
0t_gcn_wx_dense_7_biasadd_readvariableop_resourceX
Tt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resourceX
Tt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resourceT
Pt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resourceF
Bt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resourceH
Dt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resourceG
Ct_gcn_wx_model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource;
7t_gcn_wx_model_9_dense_6_matmul_readvariableop_resource<
8t_gcn_wx_model_9_dense_6_biasadd_readvariableop_resource
identity??'T-GCN-WX/dense_7/BiasAdd/ReadVariableOp?)T-GCN-WX/dense_7/Tensordot/ReadVariableOp?/T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp?.T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOp?GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?:T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?9T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?;T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?T-GCN-WX/model_9/lstm_2/while?
)T-GCN-WX/dense_7/Tensordot/ReadVariableOpReadVariableOp2t_gcn_wx_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02+
)T-GCN-WX/dense_7/Tensordot/ReadVariableOp?
T-GCN-WX/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
T-GCN-WX/dense_7/Tensordot/axes?
T-GCN-WX/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
T-GCN-WX/dense_7/Tensordot/free|
 T-GCN-WX/dense_7/Tensordot/ShapeShapeinput_13*
T0*
_output_shapes
:2"
 T-GCN-WX/dense_7/Tensordot/Shape?
(T-GCN-WX/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(T-GCN-WX/dense_7/Tensordot/GatherV2/axis?
#T-GCN-WX/dense_7/Tensordot/GatherV2GatherV2)T-GCN-WX/dense_7/Tensordot/Shape:output:0(T-GCN-WX/dense_7/Tensordot/free:output:01T-GCN-WX/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#T-GCN-WX/dense_7/Tensordot/GatherV2?
*T-GCN-WX/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*T-GCN-WX/dense_7/Tensordot/GatherV2_1/axis?
%T-GCN-WX/dense_7/Tensordot/GatherV2_1GatherV2)T-GCN-WX/dense_7/Tensordot/Shape:output:0(T-GCN-WX/dense_7/Tensordot/axes:output:03T-GCN-WX/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%T-GCN-WX/dense_7/Tensordot/GatherV2_1?
 T-GCN-WX/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 T-GCN-WX/dense_7/Tensordot/Const?
T-GCN-WX/dense_7/Tensordot/ProdProd,T-GCN-WX/dense_7/Tensordot/GatherV2:output:0)T-GCN-WX/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
T-GCN-WX/dense_7/Tensordot/Prod?
"T-GCN-WX/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"T-GCN-WX/dense_7/Tensordot/Const_1?
!T-GCN-WX/dense_7/Tensordot/Prod_1Prod.T-GCN-WX/dense_7/Tensordot/GatherV2_1:output:0+T-GCN-WX/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/dense_7/Tensordot/Prod_1?
&T-GCN-WX/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&T-GCN-WX/dense_7/Tensordot/concat/axis?
!T-GCN-WX/dense_7/Tensordot/concatConcatV2(T-GCN-WX/dense_7/Tensordot/free:output:0(T-GCN-WX/dense_7/Tensordot/axes:output:0/T-GCN-WX/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/dense_7/Tensordot/concat?
 T-GCN-WX/dense_7/Tensordot/stackPack(T-GCN-WX/dense_7/Tensordot/Prod:output:0*T-GCN-WX/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 T-GCN-WX/dense_7/Tensordot/stack?
$T-GCN-WX/dense_7/Tensordot/transpose	Transposeinput_13*T-GCN-WX/dense_7/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2&
$T-GCN-WX/dense_7/Tensordot/transpose?
"T-GCN-WX/dense_7/Tensordot/ReshapeReshape(T-GCN-WX/dense_7/Tensordot/transpose:y:0)T-GCN-WX/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2$
"T-GCN-WX/dense_7/Tensordot/Reshape?
!T-GCN-WX/dense_7/Tensordot/MatMulMatMul+T-GCN-WX/dense_7/Tensordot/Reshape:output:01T-GCN-WX/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!T-GCN-WX/dense_7/Tensordot/MatMul?
"T-GCN-WX/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"T-GCN-WX/dense_7/Tensordot/Const_2?
(T-GCN-WX/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(T-GCN-WX/dense_7/Tensordot/concat_1/axis?
#T-GCN-WX/dense_7/Tensordot/concat_1ConcatV2,T-GCN-WX/dense_7/Tensordot/GatherV2:output:0+T-GCN-WX/dense_7/Tensordot/Const_2:output:01T-GCN-WX/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#T-GCN-WX/dense_7/Tensordot/concat_1?
T-GCN-WX/dense_7/TensordotReshape+T-GCN-WX/dense_7/Tensordot/MatMul:product:0,T-GCN-WX/dense_7/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dense_7/Tensordot?
'T-GCN-WX/dense_7/BiasAdd/ReadVariableOpReadVariableOp0t_gcn_wx_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'T-GCN-WX/dense_7/BiasAdd/ReadVariableOp?
T-GCN-WX/dense_7/BiasAddBiasAdd#T-GCN-WX/dense_7/Tensordot:output:0/T-GCN-WX/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dense_7/BiasAdd?
T-GCN-WX/dropout_6/IdentityIdentity!T-GCN-WX/dense_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F2
T-GCN-WX/dropout_6/Identity?
T-GCN-WX/reshape_16/ShapeShape$T-GCN-WX/dropout_6/Identity:output:0*
T0*
_output_shapes
:2
T-GCN-WX/reshape_16/Shape?
'T-GCN-WX/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'T-GCN-WX/reshape_16/strided_slice/stack?
)T-GCN-WX/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_16/strided_slice/stack_1?
)T-GCN-WX/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)T-GCN-WX/reshape_16/strided_slice/stack_2?
!T-GCN-WX/reshape_16/strided_sliceStridedSlice"T-GCN-WX/reshape_16/Shape:output:00T-GCN-WX/reshape_16/strided_slice/stack:output:02T-GCN-WX/reshape_16/strided_slice/stack_1:output:02T-GCN-WX/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!T-GCN-WX/reshape_16/strided_slice?
#T-GCN-WX/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2%
#T-GCN-WX/reshape_16/Reshape/shape/1?
#T-GCN-WX/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#T-GCN-WX/reshape_16/Reshape/shape/2?
!T-GCN-WX/reshape_16/Reshape/shapePack*T-GCN-WX/reshape_16/strided_slice:output:0,T-GCN-WX/reshape_16/Reshape/shape/1:output:0,T-GCN-WX/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!T-GCN-WX/reshape_16/Reshape/shape?
T-GCN-WX/reshape_16/ReshapeReshape$T-GCN-WX/dropout_6/Identity:output:0*T-GCN-WX/reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
T-GCN-WX/reshape_16/Reshape?
0T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims/dim?
,T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims
ExpandDims$T-GCN-WX/reshape_16/Reshape:output:09T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2.
,T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims?
!T-GCN-WX/model_9/reshape_17/ShapeShape5T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims:output:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_9/reshape_17/Shape?
/T-GCN-WX/model_9/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_9/reshape_17/strided_slice/stack?
1T-GCN-WX/model_9/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_17/strided_slice/stack_1?
1T-GCN-WX/model_9/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_17/strided_slice/stack_2?
)T-GCN-WX/model_9/reshape_17/strided_sliceStridedSlice*T-GCN-WX/model_9/reshape_17/Shape:output:08T-GCN-WX/model_9/reshape_17/strided_slice/stack:output:0:T-GCN-WX/model_9/reshape_17/strided_slice/stack_1:output:0:T-GCN-WX/model_9/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_9/reshape_17/strided_slice?
+T-GCN-WX/model_9/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_9/reshape_17/Reshape/shape/1?
+T-GCN-WX/model_9/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+T-GCN-WX/model_9/reshape_17/Reshape/shape/2?
)T-GCN-WX/model_9/reshape_17/Reshape/shapePack2T-GCN-WX/model_9/reshape_17/strided_slice:output:04T-GCN-WX/model_9/reshape_17/Reshape/shape/1:output:04T-GCN-WX/model_9/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_9/reshape_17/Reshape/shape?
#T-GCN-WX/model_9/reshape_17/ReshapeReshape5T-GCN-WX/model_9/tf.expand_dims_4/ExpandDims:output:02T-GCN-WX/model_9/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2%
#T-GCN-WX/model_9/reshape_17/Reshape?
CT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
CT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose/perm?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose	Transpose,T-GCN-WX/model_9/reshape_17/Reshape:output:0LT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose?
:T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/ShapeShapeBT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose:y:0*
T0*
_output_shapes
:2<
:T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape?
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstackUnpackCT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2>
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack?
KT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOpReadVariableOpTt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02M
KT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOp?
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   2>
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_1?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_1UnpackET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_1?
BT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2D
BT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape/shape?
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/ReshapeReshapeBT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose:y:0KT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F2>
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape?
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpReadVariableOpTt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02Q
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2G
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/perm?
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1	TransposeWT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp:value:0NT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/perm:output:0*
T0*
_output_shapes

:FF2B
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1?
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2F
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1ReshapeDT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1:y:0MT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1?
;T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMulMatMulET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape:output:0GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F2=
;T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMul?
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2H
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1?
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2H
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2?
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shapePackET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack:output:0OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1:output:0OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2F
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2ReshapeET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMul:product:0MT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2?
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2G
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2/perm?
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2	TransposeGT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_2:output:0NT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F2B
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2?
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_2ShapeDT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0*
T0*
_output_shapes
:2>
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_2?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_2UnpackET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_2?
KT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOpReadVariableOpTt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02M
KT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOp?
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2>
<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_3?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_3UnpackET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_3?
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3ReshapeDT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0MT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3?
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpReadVariableOpTt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02Q
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2G
ET-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/perm?
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3	TransposeWT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp:value:0NT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/perm:output:0*
T0*
_output_shapes

:2B
@T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3?
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2F
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4ReshapeDT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3:y:0MT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4?
=T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMul_1MatMulGT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_3:output:0GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2?
=T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMul_1?
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2H
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1?
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2H
FT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2?
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shapePackGT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/unstack_2:output:0OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1:output:0OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2F
DT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape?
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5ReshapeGT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/MatMul_1:product:0MT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F2@
>T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5?
GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpReadVariableOpPt_gcn_wx_model_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resource*
_output_shapes

:F*
dtype02I
GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?
8T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/addAddV2GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/Reshape_5:output:0OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F2:
8T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add?
!T-GCN-WX/model_9/reshape_18/ShapeShape<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add:z:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_9/reshape_18/Shape?
/T-GCN-WX/model_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_9/reshape_18/strided_slice/stack?
1T-GCN-WX/model_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_18/strided_slice/stack_1?
1T-GCN-WX/model_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_18/strided_slice/stack_2?
)T-GCN-WX/model_9/reshape_18/strided_sliceStridedSlice*T-GCN-WX/model_9/reshape_18/Shape:output:08T-GCN-WX/model_9/reshape_18/strided_slice/stack:output:0:T-GCN-WX/model_9/reshape_18/strided_slice/stack_1:output:0:T-GCN-WX/model_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_9/reshape_18/strided_slice?
+T-GCN-WX/model_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_9/reshape_18/Reshape/shape/1?
+T-GCN-WX/model_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+T-GCN-WX/model_9/reshape_18/Reshape/shape/2?
+T-GCN-WX/model_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+T-GCN-WX/model_9/reshape_18/Reshape/shape/3?
)T-GCN-WX/model_9/reshape_18/Reshape/shapePack2T-GCN-WX/model_9/reshape_18/strided_slice:output:04T-GCN-WX/model_9/reshape_18/Reshape/shape/1:output:04T-GCN-WX/model_9/reshape_18/Reshape/shape/2:output:04T-GCN-WX/model_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_9/reshape_18/Reshape/shape?
#T-GCN-WX/model_9/reshape_18/ReshapeReshape<T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add:z:02T-GCN-WX/model_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2%
#T-GCN-WX/model_9/reshape_18/Reshape?
)T-GCN-WX/model_9/permute_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)T-GCN-WX/model_9/permute_4/transpose/perm?
$T-GCN-WX/model_9/permute_4/transpose	Transpose,T-GCN-WX/model_9/reshape_18/Reshape:output:02T-GCN-WX/model_9/permute_4/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2&
$T-GCN-WX/model_9/permute_4/transpose?
!T-GCN-WX/model_9/reshape_19/ShapeShape(T-GCN-WX/model_9/permute_4/transpose:y:0*
T0*
_output_shapes
:2#
!T-GCN-WX/model_9/reshape_19/Shape?
/T-GCN-WX/model_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_9/reshape_19/strided_slice/stack?
1T-GCN-WX/model_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_19/strided_slice/stack_1?
1T-GCN-WX/model_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1T-GCN-WX/model_9/reshape_19/strided_slice/stack_2?
)T-GCN-WX/model_9/reshape_19/strided_sliceStridedSlice*T-GCN-WX/model_9/reshape_19/Shape:output:08T-GCN-WX/model_9/reshape_19/strided_slice/stack:output:0:T-GCN-WX/model_9/reshape_19/strided_slice/stack_1:output:0:T-GCN-WX/model_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)T-GCN-WX/model_9/reshape_19/strided_slice?
+T-GCN-WX/model_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+T-GCN-WX/model_9/reshape_19/Reshape/shape/1?
+T-GCN-WX/model_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2-
+T-GCN-WX/model_9/reshape_19/Reshape/shape/2?
)T-GCN-WX/model_9/reshape_19/Reshape/shapePack2T-GCN-WX/model_9/reshape_19/strided_slice:output:04T-GCN-WX/model_9/reshape_19/Reshape/shape/1:output:04T-GCN-WX/model_9/reshape_19/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)T-GCN-WX/model_9/reshape_19/Reshape/shape?
#T-GCN-WX/model_9/reshape_19/ReshapeReshape(T-GCN-WX/model_9/permute_4/transpose:y:02T-GCN-WX/model_9/reshape_19/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2%
#T-GCN-WX/model_9/reshape_19/Reshape?
T-GCN-WX/model_9/lstm_2/ShapeShape,T-GCN-WX/model_9/reshape_19/Reshape:output:0*
T0*
_output_shapes
:2
T-GCN-WX/model_9/lstm_2/Shape?
+T-GCN-WX/model_9/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+T-GCN-WX/model_9/lstm_2/strided_slice/stack?
-T-GCN-WX/model_9/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-T-GCN-WX/model_9/lstm_2/strided_slice/stack_1?
-T-GCN-WX/model_9/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-T-GCN-WX/model_9/lstm_2/strided_slice/stack_2?
%T-GCN-WX/model_9/lstm_2/strided_sliceStridedSlice&T-GCN-WX/model_9/lstm_2/Shape:output:04T-GCN-WX/model_9/lstm_2/strided_slice/stack:output:06T-GCN-WX/model_9/lstm_2/strided_slice/stack_1:output:06T-GCN-WX/model_9/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%T-GCN-WX/model_9/lstm_2/strided_slice?
#T-GCN-WX/model_9/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#T-GCN-WX/model_9/lstm_2/zeros/mul/y?
!T-GCN-WX/model_9/lstm_2/zeros/mulMul.T-GCN-WX/model_9/lstm_2/strided_slice:output:0,T-GCN-WX/model_9/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!T-GCN-WX/model_9/lstm_2/zeros/mul?
$T-GCN-WX/model_9/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$T-GCN-WX/model_9/lstm_2/zeros/Less/y?
"T-GCN-WX/model_9/lstm_2/zeros/LessLess%T-GCN-WX/model_9/lstm_2/zeros/mul:z:0-T-GCN-WX/model_9/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"T-GCN-WX/model_9/lstm_2/zeros/Less?
&T-GCN-WX/model_9/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&T-GCN-WX/model_9/lstm_2/zeros/packed/1?
$T-GCN-WX/model_9/lstm_2/zeros/packedPack.T-GCN-WX/model_9/lstm_2/strided_slice:output:0/T-GCN-WX/model_9/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$T-GCN-WX/model_9/lstm_2/zeros/packed?
#T-GCN-WX/model_9/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#T-GCN-WX/model_9/lstm_2/zeros/Const?
T-GCN-WX/model_9/lstm_2/zerosFill-T-GCN-WX/model_9/lstm_2/zeros/packed:output:0,T-GCN-WX/model_9/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
T-GCN-WX/model_9/lstm_2/zeros?
%T-GCN-WX/model_9/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%T-GCN-WX/model_9/lstm_2/zeros_1/mul/y?
#T-GCN-WX/model_9/lstm_2/zeros_1/mulMul.T-GCN-WX/model_9/lstm_2/strided_slice:output:0.T-GCN-WX/model_9/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#T-GCN-WX/model_9/lstm_2/zeros_1/mul?
&T-GCN-WX/model_9/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&T-GCN-WX/model_9/lstm_2/zeros_1/Less/y?
$T-GCN-WX/model_9/lstm_2/zeros_1/LessLess'T-GCN-WX/model_9/lstm_2/zeros_1/mul:z:0/T-GCN-WX/model_9/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$T-GCN-WX/model_9/lstm_2/zeros_1/Less?
(T-GCN-WX/model_9/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2*
(T-GCN-WX/model_9/lstm_2/zeros_1/packed/1?
&T-GCN-WX/model_9/lstm_2/zeros_1/packedPack.T-GCN-WX/model_9/lstm_2/strided_slice:output:01T-GCN-WX/model_9/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&T-GCN-WX/model_9/lstm_2/zeros_1/packed?
%T-GCN-WX/model_9/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%T-GCN-WX/model_9/lstm_2/zeros_1/Const?
T-GCN-WX/model_9/lstm_2/zeros_1Fill/T-GCN-WX/model_9/lstm_2/zeros_1/packed:output:0.T-GCN-WX/model_9/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2!
T-GCN-WX/model_9/lstm_2/zeros_1?
&T-GCN-WX/model_9/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&T-GCN-WX/model_9/lstm_2/transpose/perm?
!T-GCN-WX/model_9/lstm_2/transpose	Transpose,T-GCN-WX/model_9/reshape_19/Reshape:output:0/T-GCN-WX/model_9/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2#
!T-GCN-WX/model_9/lstm_2/transpose?
T-GCN-WX/model_9/lstm_2/Shape_1Shape%T-GCN-WX/model_9/lstm_2/transpose:y:0*
T0*
_output_shapes
:2!
T-GCN-WX/model_9/lstm_2/Shape_1?
-T-GCN-WX/model_9/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-T-GCN-WX/model_9/lstm_2/strided_slice_1/stack?
/T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_1?
/T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_2?
'T-GCN-WX/model_9/lstm_2/strided_slice_1StridedSlice(T-GCN-WX/model_9/lstm_2/Shape_1:output:06T-GCN-WX/model_9/lstm_2/strided_slice_1/stack:output:08T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_1:output:08T-GCN-WX/model_9/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'T-GCN-WX/model_9/lstm_2/strided_slice_1?
3T-GCN-WX/model_9/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3T-GCN-WX/model_9/lstm_2/TensorArrayV2/element_shape?
%T-GCN-WX/model_9/lstm_2/TensorArrayV2TensorListReserve<T-GCN-WX/model_9/lstm_2/TensorArrayV2/element_shape:output:00T-GCN-WX/model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%T-GCN-WX/model_9/lstm_2/TensorArrayV2?
MT-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2O
MT-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
?T-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%T-GCN-WX/model_9/lstm_2/transpose:y:0VT-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?T-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor?
-T-GCN-WX/model_9/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-T-GCN-WX/model_9/lstm_2/strided_slice_2/stack?
/T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_1?
/T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_2?
'T-GCN-WX/model_9/lstm_2/strided_slice_2StridedSlice%T-GCN-WX/model_9/lstm_2/transpose:y:06T-GCN-WX/model_9/lstm_2/strided_slice_2/stack:output:08T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_1:output:08T-GCN-WX/model_9/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2)
'T-GCN-WX/model_9/lstm_2/strided_slice_2?
9T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpBt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02;
9T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?
*T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMulMatMul0T-GCN-WX/model_9/lstm_2/strided_slice_2:output:0AT-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul?
;T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpDt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?
,T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1MatMul&T-GCN-WX/model_9/lstm_2/zeros:output:0CT-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1?
'T-GCN-WX/model_9/lstm_2/lstm_cell_2/addAddV24T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul:product:06T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2)
'T-GCN-WX/model_9/lstm_2/lstm_cell_2/add?
:T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpCt_gcn_wx_model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?
+T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAddBiasAdd+T-GCN-WX/model_9/lstm_2/lstm_cell_2/add:z:0BT-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd?
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/Const?
3T-GCN-WX/model_9/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3T-GCN-WX/model_9/lstm_2/lstm_cell_2/split/split_dim?
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/splitSplit<T-GCN-WX/model_9/lstm_2/lstm_cell_2/split/split_dim:output:04T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2+
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/split?
+T-GCN-WX/model_9/lstm_2/lstm_cell_2/SigmoidSigmoid2T-GCN-WX/model_9/lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2-
+T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid?
-T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_1Sigmoid2T-GCN-WX/model_9/lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2/
-T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_1?
'T-GCN-WX/model_9/lstm_2/lstm_cell_2/mulMul1T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_1:y:0(T-GCN-WX/model_9/lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:??????????2)
'T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul?
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul_1Mul/T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid:y:02T-GCN-WX/model_9/lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2+
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul_1?
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/add_1AddV2+T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul:z:0-T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2+
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/add_1?
-T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_2Sigmoid2T-GCN-WX/model_9/lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2/
-T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_2?
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul_2Mul1T-GCN-WX/model_9/lstm_2/lstm_cell_2/Sigmoid_2:y:0-T-GCN-WX/model_9/lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)T-GCN-WX/model_9/lstm_2/lstm_cell_2/mul_2?
5T-GCN-WX/model_9/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5T-GCN-WX/model_9/lstm_2/TensorArrayV2_1/element_shape?
'T-GCN-WX/model_9/lstm_2/TensorArrayV2_1TensorListReserve>T-GCN-WX/model_9/lstm_2/TensorArrayV2_1/element_shape:output:00T-GCN-WX/model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'T-GCN-WX/model_9/lstm_2/TensorArrayV2_1~
T-GCN-WX/model_9/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
T-GCN-WX/model_9/lstm_2/time?
0T-GCN-WX/model_9/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0T-GCN-WX/model_9/lstm_2/while/maximum_iterations?
*T-GCN-WX/model_9/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*T-GCN-WX/model_9/lstm_2/while/loop_counter?
T-GCN-WX/model_9/lstm_2/whileWhile3T-GCN-WX/model_9/lstm_2/while/loop_counter:output:09T-GCN-WX/model_9/lstm_2/while/maximum_iterations:output:0%T-GCN-WX/model_9/lstm_2/time:output:00T-GCN-WX/model_9/lstm_2/TensorArrayV2_1:handle:0&T-GCN-WX/model_9/lstm_2/zeros:output:0(T-GCN-WX/model_9/lstm_2/zeros_1:output:00T-GCN-WX/model_9/lstm_2/strided_slice_1:output:0OT-GCN-WX/model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resourceDt_gcn_wx_model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resourceCt_gcn_wx_model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*4
body,R*
(T-GCN-WX_model_9_lstm_2_while_body_54712*4
cond,R*
(T-GCN-WX_model_9_lstm_2_while_cond_54711*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
T-GCN-WX/model_9/lstm_2/while?
HT-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2J
HT-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape?
:T-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack&T-GCN-WX/model_9/lstm_2/while:output:3QT-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02<
:T-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack?
-T-GCN-WX/model_9/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-T-GCN-WX/model_9/lstm_2/strided_slice_3/stack?
/T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_1?
/T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_2?
'T-GCN-WX/model_9/lstm_2/strided_slice_3StridedSliceCT-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:06T-GCN-WX/model_9/lstm_2/strided_slice_3/stack:output:08T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_1:output:08T-GCN-WX/model_9/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2)
'T-GCN-WX/model_9/lstm_2/strided_slice_3?
(T-GCN-WX/model_9/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(T-GCN-WX/model_9/lstm_2/transpose_1/perm?
#T-GCN-WX/model_9/lstm_2/transpose_1	TransposeCT-GCN-WX/model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:01T-GCN-WX/model_9/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2%
#T-GCN-WX/model_9/lstm_2/transpose_1?
T-GCN-WX/model_9/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
T-GCN-WX/model_9/lstm_2/runtime?
#T-GCN-WX/model_9/dropout_7/IdentityIdentity0T-GCN-WX/model_9/lstm_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2%
#T-GCN-WX/model_9/dropout_7/Identity?
.T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOpReadVariableOp7t_gcn_wx_model_9_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype020
.T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOp?
T-GCN-WX/model_9/dense_6/MatMulMatMul,T-GCN-WX/model_9/dropout_7/Identity:output:06T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2!
T-GCN-WX/model_9/dense_6/MatMul?
/T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOpReadVariableOp8t_gcn_wx_model_9_dense_6_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype021
/T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp?
 T-GCN-WX/model_9/dense_6/BiasAddBiasAdd)T-GCN-WX/model_9/dense_6/MatMul:product:07T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2"
 T-GCN-WX/model_9/dense_6/BiasAdd?
 T-GCN-WX/model_9/dense_6/SigmoidSigmoid)T-GCN-WX/model_9/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2"
 T-GCN-WX/model_9/dense_6/Sigmoid?
IdentityIdentity$T-GCN-WX/model_9/dense_6/Sigmoid:y:0(^T-GCN-WX/dense_7/BiasAdd/ReadVariableOp*^T-GCN-WX/dense_7/Tensordot/ReadVariableOp0^T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp/^T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOpH^T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpP^T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpP^T-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp;^T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:^T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp<^T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^T-GCN-WX/model_9/lstm_2/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2R
'T-GCN-WX/dense_7/BiasAdd/ReadVariableOp'T-GCN-WX/dense_7/BiasAdd/ReadVariableOp2V
)T-GCN-WX/dense_7/Tensordot/ReadVariableOp)T-GCN-WX/dense_7/Tensordot/ReadVariableOp2b
/T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp/T-GCN-WX/model_9/dense_6/BiasAdd/ReadVariableOp2`
.T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOp.T-GCN-WX/model_9/dense_6/MatMul/ReadVariableOp2?
GT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpGT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp2?
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpOT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp2?
OT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpOT-GCN-WX/model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2x
:T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:T-GCN-WX/model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2v
9T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp9T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp2z
;T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp;T-GCN-WX/model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2>
T-GCN-WX/model_9/lstm_2/whileT-GCN-WX/model_9/lstm_2/while:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_13
?
F
*__inference_reshape_19_layer_call_fn_57755

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
GPU 2J 8? *N
fIRG
E__inference_reshape_19_layer_call_and_return_conditional_losses_555582
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
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_55910

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
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_56118

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
?@
?
while_body_55631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
a
E__inference_reshape_17_layer_call_and_return_conditional_losses_55441

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
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_58407

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
a
E__inference_reshape_16_layer_call_and_return_conditional_losses_56177

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
A__inference_lstm_2_layer_call_and_return_conditional_losses_58053

inputs.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_57970*
condR
while_cond_57969*M
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
:??????????*
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
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_58412

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
?
?
(__inference_T-GCN-WX_layer_call_fn_57027

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
GPU 2J 8? *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_563572
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
?
?
while_cond_55211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55211___redundant_placeholder03
/while_while_cond_55211___redundant_placeholder13
/while_while_cond_55211___redundant_placeholder23
/while_while_cond_55211___redundant_placeholder3
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
?
|
'__inference_dense_7_layer_call_fn_57066

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
GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_561182
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
?
F
*__inference_reshape_18_layer_call_fn_57737

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
GPU 2J 8? *N
fIRG
E__inference_reshape_18_layer_call_and_return_conditional_losses_555362
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
?
?
'__inference_model_9_layer_call_fn_56084
input_14
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560652
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
input_14
?X
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_55863

inputs.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55780*
condR
while_cond_55779*M
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
:??????????*
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
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
`
D__inference_permute_4_layer_call_and_return_conditional_losses_54810

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
?
F
*__inference_reshape_17_layer_call_fn_57654

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
GPU 2J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_554412
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
?@
?
while_body_55780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
?$
?
while_body_55344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_2_55368_0
while_lstm_cell_2_55370_0
while_lstm_cell_2_55372_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_2_55368
while_lstm_cell_2_55370
while_lstm_cell_2_55372??)while/lstm_cell_2/StatefulPartitionedCall?
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
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_55368_0while_lstm_cell_2_55370_0while_lstm_cell_2_55372_0*
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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_549182+
)while/lstm_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_55368while_lstm_cell_2_55368_0"4
while_lstm_cell_2_55370while_lstm_cell_2_55370_0"4
while_lstm_cell_2_55372while_lstm_cell_2_55372_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 
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
?
while_cond_58140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58140___redundant_placeholder03
/while_while_cond_58140___redundant_placeholder13
/while_while_cond_58140___redundant_placeholder23
/while_while_cond_58140___redundant_placeholder3
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
?
B__inference_dense_6_layer_call_and_return_conditional_losses_58433

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
?'
?
B__inference_model_9_layer_call_and_return_conditional_losses_56014

inputs-
)fixed_adjacency_graph_convolution_2_55990-
)fixed_adjacency_graph_convolution_2_55992-
)fixed_adjacency_graph_convolution_2_55994
lstm_2_56000
lstm_2_56002
lstm_2_56004
dense_6_56008
dense_6_56010
identity??dense_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinputs(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDims?
reshape_17/PartitionedCallPartitionedCall$tf.expand_dims_4/ExpandDims:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_554412
reshape_17/PartitionedCall?
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_17/PartitionedCall:output:0)fixed_adjacency_graph_convolution_2_55990)fixed_adjacency_graph_convolution_2_55992)fixed_adjacency_graph_convolution_2_55994*
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
GPU 2J 8? *g
fbR`
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_555022=
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_18_layer_call_and_return_conditional_losses_555362
reshape_18/PartitionedCall?
permute_4/PartitionedCallPartitionedCall#reshape_18/PartitionedCall:output:0*
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
D__inference_permute_4_layer_call_and_return_conditional_losses_548102
permute_4/PartitionedCall?
reshape_19/PartitionedCallPartitionedCall"permute_4/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_19_layer_call_and_return_conditional_losses_555582
reshape_19/PartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_19/PartitionedCall:output:0lstm_2_56000lstm_2_56002lstm_2_56004*
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_557142 
lstm_2/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559052#
!dropout_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_6_56008dense_6_56010*
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
GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_559342!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_2/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
a
E__inference_reshape_16_layer_call_and_return_conditional_losses_57106

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
?
B__inference_dense_6_layer_call_and_return_conditional_losses_55934

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
?
?
while_cond_58289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58289___redundant_placeholder03
/while_while_cond_58289___redundant_placeholder13
/while_while_cond_58289___redundant_placeholder23
/while_while_cond_58289___redundant_placeholder3
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
?
'__inference_model_9_layer_call_fn_57636

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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560652
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_55281

inputs
lstm_cell_2_55199
lstm_cell_2_55201
lstm_cell_2_55203
identity??#lstm_cell_2/StatefulPartitionedCall?whileD
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
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_55199lstm_cell_2_55201lstm_cell_2_55203*
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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_548872%
#lstm_cell_2/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_55199lstm_cell_2_55201lstm_cell_2_55203*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55212*
condR
while_cond_55211*M
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????F
 
_user_specified_nameinputs
?
?
while_cond_55343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55343___redundant_placeholder03
/while_while_cond_55343___redundant_placeholder13
/while_while_cond_55343___redundant_placeholder23
/while_while_cond_55343___redundant_placeholder3
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
?
?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56245
input_13
dense_7_56129
dense_7_56131
model_9_56227
model_9_56229
model_9_56231
model_9_56233
model_9_56235
model_9_56237
model_9_56239
model_9_56241
identity??dense_7/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?model_9/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_7_56129dense_7_56131*
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
GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_561182!
dense_7/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_561462#
!dropout_6/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_16_layer_call_and_return_conditional_losses_561772
reshape_16/PartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0model_9_56227model_9_56229model_9_56231model_9_56233model_9_56235model_9_56237model_9_56239model_9_56241*
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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560142!
model_9/StatefulPartitionedCall?
IdentityIdentity(model_9/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall ^model_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????F
"
_user_specified_name
input_13
?Y
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58224
inputs_0.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileF
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_58141*
condR
while_cond_58140*M
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
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?@
?
while_body_57821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
a
E__inference_reshape_18_layer_call_and_return_conditional_losses_55536

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
?Y
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58373
inputs_0.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileF
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_58290*
condR
while_cond_58289*M
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
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????F
"
_user_specified_name
inputs/0
?
?
while_cond_57969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_57969___redundant_placeholder03
/while_while_cond_57969___redundant_placeholder13
/while_while_cond_57969___redundant_placeholder23
/while_while_cond_57969___redundant_placeholder3
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
?J
?	
lstm_2_while_body_57503*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0?
;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0>
:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor;
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource=
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource<
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource??/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2@
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????F*
element_dtype022
0lstm_2/while/TensorArrayV2Read/TensorListGetItem?
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype020
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp?
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_2/while/lstm_cell_2/MatMul?
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp?
!lstm_2/while/lstm_cell_2/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_2/while/lstm_cell_2/MatMul_1?
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/MatMul:product:0+lstm_2/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_2/while/lstm_cell_2/add?
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp?
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd lstm_2/while/lstm_cell_2/add:z:07lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_2/while/lstm_cell_2/BiasAdd?
lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_2/while/lstm_cell_2/Const?
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_2/while/lstm_cell_2/split/split_dim?
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:0)lstm_2/while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2 
lstm_2/while/lstm_cell_2/split?
 lstm_2/while/lstm_cell_2/SigmoidSigmoid'lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_2/while/lstm_cell_2/Sigmoid?
"lstm_2/while/lstm_cell_2/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_2/while/lstm_cell_2/Sigmoid_1?
lstm_2/while/lstm_cell_2/mulMul&lstm_2/while/lstm_cell_2/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*(
_output_shapes
:??????????2
lstm_2/while/lstm_cell_2/mul?
lstm_2/while/lstm_cell_2/mul_1Mul$lstm_2/while/lstm_cell_2/Sigmoid:y:0'lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/mul_1?
lstm_2/while/lstm_cell_2/add_1AddV2 lstm_2/while/lstm_cell_2/mul:z:0"lstm_2/while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/add_1?
"lstm_2/while/lstm_cell_2/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_2/while/lstm_cell_2/Sigmoid_2?
lstm_2/while/lstm_cell_2/mul_2Mul&lstm_2/while/lstm_cell_2/Sigmoid_2:y:0"lstm_2/while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_2/while/lstm_cell_2/mul_2?
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_2/while/TensorArrayV2Write/TensorListSetItemj
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add/y?
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/addn
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add_1/y?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/add_1?
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations0^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_1?
lstm_2/while/Identity_2Identitylstm_2/while/add:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_2?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_3?
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_2:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_2/while/Identity_4?
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_1:z:00^lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
lstm_2/while/Identity_5"7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_2_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_2_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_2_matmul_readvariableop_resource9lstm_2_while_lstm_cell_2_matmul_readvariableop_resource_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2b
/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_2/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_2/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
'__inference_model_9_layer_call_fn_57615

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
GPU 2J 8? *K
fFRD
B__inference_model_9_layer_call_and_return_conditional_losses_560142
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
??
?	
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56977

inputs-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resourceO
Kmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resourceO
Kmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resourceK
Gmodel_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resource=
9model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource?
;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource>
:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource2
.model_9_dense_6_matmul_readvariableop_resource3
/model_9_dense_6_biasadd_readvariableop_resource
identity??dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?&model_9/dense_6/BiasAdd/ReadVariableOp?%model_9/dense_6/MatMul/ReadVariableOp?>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?model_9/lstm_2/while?
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes?
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_7/Tensordot/freeh
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape?
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2?
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod?
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1?
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack?
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????F2
dense_7/Tensordot/transpose?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_7/Tensordot/Reshape?
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/Tensordot/MatMul?
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2?
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????F2
dense_7/Tensordot?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????F2
dense_7/BiasAdd?
dropout_6/IdentityIdentitydense_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????F2
dropout_6/Identityo
reshape_16/ShapeShapedropout_6/Identity:output:0*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapedropout_6/Identity:output:0!reshape_16/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
reshape_16/Reshape?
'model_9/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/tf.expand_dims_4/ExpandDims/dim?
#model_9/tf.expand_dims_4/ExpandDims
ExpandDimsreshape_16/Reshape:output:00model_9/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2%
#model_9/tf.expand_dims_4/ExpandDims?
model_9/reshape_17/ShapeShape,model_9/tf.expand_dims_4/ExpandDims:output:0*
T0*
_output_shapes
:2
model_9/reshape_17/Shape?
&model_9/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_17/strided_slice/stack?
(model_9/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_17/strided_slice/stack_1?
(model_9/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_17/strided_slice/stack_2?
 model_9/reshape_17/strided_sliceStridedSlice!model_9/reshape_17/Shape:output:0/model_9/reshape_17/strided_slice/stack:output:01model_9/reshape_17/strided_slice/stack_1:output:01model_9/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_17/strided_slice?
"model_9/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_17/Reshape/shape/1?
"model_9/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/reshape_17/Reshape/shape/2?
 model_9/reshape_17/Reshape/shapePack)model_9/reshape_17/strided_slice:output:0+model_9/reshape_17/Reshape/shape/1:output:0+model_9/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_17/Reshape/shape?
model_9/reshape_17/ReshapeReshape,model_9/tf.expand_dims_4/ExpandDims:output:0)model_9/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_9/reshape_17/Reshape?
:model_9/fixed_adjacency_graph_convolution_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2<
:model_9/fixed_adjacency_graph_convolution_2/transpose/perm?
5model_9/fixed_adjacency_graph_convolution_2/transpose	Transpose#model_9/reshape_17/Reshape:output:0Cmodel_9/fixed_adjacency_graph_convolution_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/transpose?
1model_9/fixed_adjacency_graph_convolution_2/ShapeShape9model_9/fixed_adjacency_graph_convolution_2/transpose:y:0*
T0*
_output_shapes
:23
1model_9/fixed_adjacency_graph_convolution_2/Shape?
3model_9/fixed_adjacency_graph_convolution_2/unstackUnpack:model_9/fixed_adjacency_graph_convolution_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num25
3model_9/fixed_adjacency_graph_convolution_2/unstack?
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02D
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_1/ReadVariableOp?
3model_9/fixed_adjacency_graph_convolution_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"F   F   25
3model_9/fixed_adjacency_graph_convolution_2/Shape_1?
5model_9/fixed_adjacency_graph_convolution_2/unstack_1Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_1?
9model_9/fixed_adjacency_graph_convolution_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2;
9model_9/fixed_adjacency_graph_convolution_2/Reshape/shape?
3model_9/fixed_adjacency_graph_convolution_2/ReshapeReshape9model_9/fixed_adjacency_graph_convolution_2/transpose:y:0Bmodel_9/fixed_adjacency_graph_convolution_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????F25
3model_9/fixed_adjacency_graph_convolution_2/Reshape?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_1_readvariableop_resource*
_output_shapes

:FF*
dtype02H
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp?
<model_9/fixed_adjacency_graph_convolution_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_1/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_1	TransposeNmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp:value:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_1/perm:output:0*
T0*
_output_shapes

:FF29
7model_9/fixed_adjacency_graph_convolution_2/transpose_1?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"F   ????2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_1Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_1:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:FF27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_1?
2model_9/fixed_adjacency_graph_convolution_2/MatMulMatMul<model_9/fixed_adjacency_graph_convolution_2/Reshape:output:0>model_9/fixed_adjacency_graph_convolution_2/Reshape_1:output:0*
T0*'
_output_shapes
:?????????F24
2model_9/fixed_adjacency_graph_convolution_2/MatMul?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shapePack<model_9/fixed_adjacency_graph_convolution_2/unstack:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/1:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_2Reshape<model_9/fixed_adjacency_graph_convolution_2/MatMul:product:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_2?
<model_9/fixed_adjacency_graph_convolution_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_2/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_2	Transpose>model_9/fixed_adjacency_graph_convolution_2/Reshape_2:output:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????F29
7model_9/fixed_adjacency_graph_convolution_2/transpose_2?
3model_9/fixed_adjacency_graph_convolution_2/Shape_2Shape;model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0*
T0*
_output_shapes
:25
3model_9/fixed_adjacency_graph_convolution_2/Shape_2?
5model_9/fixed_adjacency_graph_convolution_2/unstack_2Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_2?
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02D
Bmodel_9/fixed_adjacency_graph_convolution_2/Shape_3/ReadVariableOp?
3model_9/fixed_adjacency_graph_convolution_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      25
3model_9/fixed_adjacency_graph_convolution_2/Shape_3?
5model_9/fixed_adjacency_graph_convolution_2/unstack_3Unpack<model_9/fixed_adjacency_graph_convolution_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num27
5model_9/fixed_adjacency_graph_convolution_2/unstack_3?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_3Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_2:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_3?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpReadVariableOpKmodel_9_fixed_adjacency_graph_convolution_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02H
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp?
<model_9/fixed_adjacency_graph_convolution_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<model_9/fixed_adjacency_graph_convolution_2/transpose_3/perm?
7model_9/fixed_adjacency_graph_convolution_2/transpose_3	TransposeNmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp:value:0Emodel_9/fixed_adjacency_graph_convolution_2/transpose_3/perm:output:0*
T0*
_output_shapes

:29
7model_9/fixed_adjacency_graph_convolution_2/transpose_3?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_4Reshape;model_9/fixed_adjacency_graph_convolution_2/transpose_3:y:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_4?
4model_9/fixed_adjacency_graph_convolution_2/MatMul_1MatMul>model_9/fixed_adjacency_graph_convolution_2/Reshape_3:output:0>model_9/fixed_adjacency_graph_convolution_2/Reshape_4:output:0*
T0*'
_output_shapes
:?????????26
4model_9/fixed_adjacency_graph_convolution_2/MatMul_1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2?
;model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shapePack>model_9/fixed_adjacency_graph_convolution_2/unstack_2:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/1:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;model_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape?
5model_9/fixed_adjacency_graph_convolution_2/Reshape_5Reshape>model_9/fixed_adjacency_graph_convolution_2/MatMul_1:product:0Dmodel_9/fixed_adjacency_graph_convolution_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????F27
5model_9/fixed_adjacency_graph_convolution_2/Reshape_5?
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpReadVariableOpGmodel_9_fixed_adjacency_graph_convolution_2_add_readvariableop_resource*
_output_shapes

:F*
dtype02@
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp?
/model_9/fixed_adjacency_graph_convolution_2/addAddV2>model_9/fixed_adjacency_graph_convolution_2/Reshape_5:output:0Fmodel_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????F21
/model_9/fixed_adjacency_graph_convolution_2/add?
model_9/reshape_18/ShapeShape3model_9/fixed_adjacency_graph_convolution_2/add:z:0*
T0*
_output_shapes
:2
model_9/reshape_18/Shape?
&model_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_18/strided_slice/stack?
(model_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_18/strided_slice/stack_1?
(model_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_18/strided_slice/stack_2?
 model_9/reshape_18/strided_sliceStridedSlice!model_9/reshape_18/Shape:output:0/model_9/reshape_18/strided_slice/stack:output:01model_9/reshape_18/strided_slice/stack_1:output:01model_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_18/strided_slice?
"model_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_18/Reshape/shape/1?
"model_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_9/reshape_18/Reshape/shape/2?
"model_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/reshape_18/Reshape/shape/3?
 model_9/reshape_18/Reshape/shapePack)model_9/reshape_18/strided_slice:output:0+model_9/reshape_18/Reshape/shape/1:output:0+model_9/reshape_18/Reshape/shape/2:output:0+model_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_18/Reshape/shape?
model_9/reshape_18/ReshapeReshape3model_9/fixed_adjacency_graph_convolution_2/add:z:0)model_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????F2
model_9/reshape_18/Reshape?
 model_9/permute_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 model_9/permute_4/transpose/perm?
model_9/permute_4/transpose	Transpose#model_9/reshape_18/Reshape:output:0)model_9/permute_4/transpose/perm:output:0*
T0*/
_output_shapes
:?????????F2
model_9/permute_4/transpose?
model_9/reshape_19/ShapeShapemodel_9/permute_4/transpose:y:0*
T0*
_output_shapes
:2
model_9/reshape_19/Shape?
&model_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/reshape_19/strided_slice/stack?
(model_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_19/strided_slice/stack_1?
(model_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_9/reshape_19/strided_slice/stack_2?
 model_9/reshape_19/strided_sliceStridedSlice!model_9/reshape_19/Shape:output:0/model_9/reshape_19/strided_slice/stack:output:01model_9/reshape_19/strided_slice/stack_1:output:01model_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_9/reshape_19/strided_slice?
"model_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_9/reshape_19/Reshape/shape/1?
"model_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :F2$
"model_9/reshape_19/Reshape/shape/2?
 model_9/reshape_19/Reshape/shapePack)model_9/reshape_19/strided_slice:output:0+model_9/reshape_19/Reshape/shape/1:output:0+model_9/reshape_19/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_9/reshape_19/Reshape/shape?
model_9/reshape_19/ReshapeReshapemodel_9/permute_4/transpose:y:0)model_9/reshape_19/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????F2
model_9/reshape_19/Reshape
model_9/lstm_2/ShapeShape#model_9/reshape_19/Reshape:output:0*
T0*
_output_shapes
:2
model_9/lstm_2/Shape?
"model_9/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_9/lstm_2/strided_slice/stack?
$model_9/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/lstm_2/strided_slice/stack_1?
$model_9/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_9/lstm_2/strided_slice/stack_2?
model_9/lstm_2/strided_sliceStridedSlicemodel_9/lstm_2/Shape:output:0+model_9/lstm_2/strided_slice/stack:output:0-model_9/lstm_2/strided_slice/stack_1:output:0-model_9/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_2/strided_slice{
model_9/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/mul/y?
model_9/lstm_2/zeros/mulMul%model_9/lstm_2/strided_slice:output:0#model_9/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros/mul}
model_9/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/Less/y?
model_9/lstm_2/zeros/LessLessmodel_9/lstm_2/zeros/mul:z:0$model_9/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros/Less?
model_9/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros/packed/1?
model_9/lstm_2/zeros/packedPack%model_9/lstm_2/strided_slice:output:0&model_9/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_2/zeros/packed}
model_9/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/zeros/Const?
model_9/lstm_2/zerosFill$model_9/lstm_2/zeros/packed:output:0#model_9/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_9/lstm_2/zeros
model_9/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros_1/mul/y?
model_9/lstm_2/zeros_1/mulMul%model_9/lstm_2/strided_slice:output:0%model_9/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros_1/mul?
model_9/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_9/lstm_2/zeros_1/Less/y?
model_9/lstm_2/zeros_1/LessLessmodel_9/lstm_2/zeros_1/mul:z:0&model_9/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_2/zeros_1/Less?
model_9/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
model_9/lstm_2/zeros_1/packed/1?
model_9/lstm_2/zeros_1/packedPack%model_9/lstm_2/strided_slice:output:0(model_9/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_2/zeros_1/packed?
model_9/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/zeros_1/Const?
model_9/lstm_2/zeros_1Fill&model_9/lstm_2/zeros_1/packed:output:0%model_9/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_9/lstm_2/zeros_1?
model_9/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_9/lstm_2/transpose/perm?
model_9/lstm_2/transpose	Transpose#model_9/reshape_19/Reshape:output:0&model_9/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????F2
model_9/lstm_2/transpose|
model_9/lstm_2/Shape_1Shapemodel_9/lstm_2/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_2/Shape_1?
$model_9/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/lstm_2/strided_slice_1/stack?
&model_9/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_1/stack_1?
&model_9/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_1/stack_2?
model_9/lstm_2/strided_slice_1StridedSlicemodel_9/lstm_2/Shape_1:output:0-model_9/lstm_2/strided_slice_1/stack:output:0/model_9/lstm_2/strided_slice_1/stack_1:output:0/model_9/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
model_9/lstm_2/strided_slice_1?
*model_9/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_9/lstm_2/TensorArrayV2/element_shape?
model_9/lstm_2/TensorArrayV2TensorListReserve3model_9/lstm_2/TensorArrayV2/element_shape:output:0'model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_2/TensorArrayV2?
Dmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????F   2F
Dmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
6model_9/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_2/transpose:y:0Mmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_9/lstm_2/TensorArrayUnstack/TensorListFromTensor?
$model_9/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_9/lstm_2/strided_slice_2/stack?
&model_9/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_2/stack_1?
&model_9/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_2/stack_2?
model_9/lstm_2/strided_slice_2StridedSlicemodel_9/lstm_2/transpose:y:0-model_9/lstm_2/strided_slice_2/stack:output:0/model_9/lstm_2/strided_slice_2/stack_1:output:0/model_9/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????F*
shrink_axis_mask2 
model_9/lstm_2/strided_slice_2?
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp9model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype022
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp?
!model_9/lstm_2/lstm_cell_2/MatMulMatMul'model_9/lstm_2/strided_slice_2:output:08model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_9/lstm_2/lstm_cell_2/MatMul?
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp?
#model_9/lstm_2/lstm_cell_2/MatMul_1MatMulmodel_9/lstm_2/zeros:output:0:model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_9/lstm_2/lstm_cell_2/MatMul_1?
model_9/lstm_2/lstm_cell_2/addAddV2+model_9/lstm_2/lstm_cell_2/MatMul:product:0-model_9/lstm_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
model_9/lstm_2/lstm_cell_2/add?
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp?
"model_9/lstm_2/lstm_cell_2/BiasAddBiasAdd"model_9/lstm_2/lstm_cell_2/add:z:09model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_9/lstm_2/lstm_cell_2/BiasAdd?
 model_9/lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/lstm_2/lstm_cell_2/Const?
*model_9/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_9/lstm_2/lstm_cell_2/split/split_dim?
 model_9/lstm_2/lstm_cell_2/splitSplit3model_9/lstm_2/lstm_cell_2/split/split_dim:output:0+model_9/lstm_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 model_9/lstm_2/lstm_cell_2/split?
"model_9/lstm_2/lstm_cell_2/SigmoidSigmoid)model_9/lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2$
"model_9/lstm_2/lstm_cell_2/Sigmoid?
$model_9/lstm_2/lstm_cell_2/Sigmoid_1Sigmoid)model_9/lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/lstm_cell_2/Sigmoid_1?
model_9/lstm_2/lstm_cell_2/mulMul(model_9/lstm_2/lstm_cell_2/Sigmoid_1:y:0model_9/lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:??????????2 
model_9/lstm_2/lstm_cell_2/mul?
 model_9/lstm_2/lstm_cell_2/mul_1Mul&model_9/lstm_2/lstm_cell_2/Sigmoid:y:0)model_9/lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/mul_1?
 model_9/lstm_2/lstm_cell_2/add_1AddV2"model_9/lstm_2/lstm_cell_2/mul:z:0$model_9/lstm_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/add_1?
$model_9/lstm_2/lstm_cell_2/Sigmoid_2Sigmoid)model_9/lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2&
$model_9/lstm_2/lstm_cell_2/Sigmoid_2?
 model_9/lstm_2/lstm_cell_2/mul_2Mul(model_9/lstm_2/lstm_cell_2/Sigmoid_2:y:0$model_9/lstm_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2"
 model_9/lstm_2/lstm_cell_2/mul_2?
,model_9/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2.
,model_9/lstm_2/TensorArrayV2_1/element_shape?
model_9/lstm_2/TensorArrayV2_1TensorListReserve5model_9/lstm_2/TensorArrayV2_1/element_shape:output:0'model_9/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_9/lstm_2/TensorArrayV2_1l
model_9/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_2/time?
'model_9/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/lstm_2/while/maximum_iterations?
!model_9/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_9/lstm_2/while/loop_counter?
model_9/lstm_2/whileWhile*model_9/lstm_2/while/loop_counter:output:00model_9/lstm_2/while/maximum_iterations:output:0model_9/lstm_2/time:output:0'model_9/lstm_2/TensorArrayV2_1:handle:0model_9/lstm_2/zeros:output:0model_9/lstm_2/zeros_1:output:0'model_9/lstm_2/strided_slice_1:output:0Fmodel_9/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_9_lstm_2_lstm_cell_2_matmul_readvariableop_resource;model_9_lstm_2_lstm_cell_2_matmul_1_readvariableop_resource:model_9_lstm_2_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_9_lstm_2_while_body_56886*+
cond#R!
model_9_lstm_2_while_cond_56885*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
model_9/lstm_2/while?
?model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?model_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape?
1model_9/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_2/while:output:3Hmodel_9/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype023
1model_9/lstm_2/TensorArrayV2Stack/TensorListStack?
$model_9/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$model_9/lstm_2/strided_slice_3/stack?
&model_9/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_9/lstm_2/strided_slice_3/stack_1?
&model_9/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_9/lstm_2/strided_slice_3/stack_2?
model_9/lstm_2/strided_slice_3StridedSlice:model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-model_9/lstm_2/strided_slice_3/stack:output:0/model_9/lstm_2/strided_slice_3/stack_1:output:0/model_9/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2 
model_9/lstm_2/strided_slice_3?
model_9/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_9/lstm_2/transpose_1/perm?
model_9/lstm_2/transpose_1	Transpose:model_9/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0(model_9/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
model_9/lstm_2/transpose_1?
model_9/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_2/runtime?
model_9/dropout_7/IdentityIdentity'model_9/lstm_2/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
model_9/dropout_7/Identity?
%model_9/dense_6/MatMul/ReadVariableOpReadVariableOp.model_9_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?F*
dtype02'
%model_9/dense_6/MatMul/ReadVariableOp?
model_9/dense_6/MatMulMatMul#model_9/dropout_7/Identity:output:0-model_9/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/MatMul?
&model_9/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_9_dense_6_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_9/dense_6/BiasAdd/ReadVariableOp?
model_9/dense_6/BiasAddBiasAdd model_9/dense_6/MatMul:product:0.model_9/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/BiasAdd?
model_9/dense_6/SigmoidSigmoid model_9/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????F2
model_9/dense_6/Sigmoid?
IdentityIdentitymodel_9/dense_6/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp'^model_9/dense_6/BiasAdd/ReadVariableOp&^model_9/dense_6/MatMul/ReadVariableOp?^model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOpG^model_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpG^model_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2^model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp1^model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp3^model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp^model_9/lstm_2/while*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????F::::::::::2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2P
&model_9/dense_6/BiasAdd/ReadVariableOp&model_9/dense_6/BiasAdd/ReadVariableOp2N
%model_9/dense_6/MatMul/ReadVariableOp%model_9/dense_6/MatMul/ReadVariableOp2?
>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp>model_9/fixed_adjacency_graph_convolution_2/add/ReadVariableOp2?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOpFmodel_9/fixed_adjacency_graph_convolution_2/transpose_1/ReadVariableOp2?
Fmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOpFmodel_9/fixed_adjacency_graph_convolution_2/transpose_3/ReadVariableOp2f
1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp1model_9/lstm_2/lstm_cell_2/BiasAdd/ReadVariableOp2d
0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp0model_9/lstm_2/lstm_cell_2/MatMul/ReadVariableOp2h
2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2model_9/lstm_2/lstm_cell_2/MatMul_1/ReadVariableOp2,
model_9/lstm_2/whilemodel_9/lstm_2/while:W S
/
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
a
E__inference_reshape_19_layer_call_and_return_conditional_losses_57750

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
?X
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57904

inputs.
*lstm_cell_2_matmul_readvariableop_resource0
,lstm_cell_2_matmul_1_readvariableop_resource/
+lstm_cell_2_biasadd_readvariableop_resource
identity??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	F?*
dtype02#
!lstm_cell_2/MatMul/ReadVariableOp?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul?
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#lstm_cell_2/MatMul_1/ReadVariableOp?
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/MatMul_1?
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add?
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_2/BiasAdd/ReadVariableOp?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/BiasAddh
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/Const|
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_2/split/split_dim?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_2/split?
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid?
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_1?
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul?
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_1?
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/add_1?
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_2/Sigmoid_2?
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_2/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_57821*
condR
while_cond_57820*M
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
:??????????*
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
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
b
)__inference_dropout_7_layer_call_fn_58417

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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559052
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
?
E
)__inference_dropout_7_layer_call_fn_58422

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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559102
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
?$
?
while_body_55212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_2_55236_0
while_lstm_cell_2_55238_0
while_lstm_cell_2_55240_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_2_55236
while_lstm_cell_2_55238
while_lstm_cell_2_55240??)while/lstm_cell_2/StatefulPartitionedCall?
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
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_55236_0while_lstm_cell_2_55238_0while_lstm_cell_2_55240_0*
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
GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_548872+
)while/lstm_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2*^while/lstm_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_55236while_lstm_cell_2_55236_0"4
while_lstm_cell_2_55238while_lstm_cell_2_55238_0"4
while_lstm_cell_2_55240while_lstm_cell_2_55240_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 
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
?,
?
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_57707
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
?&
?
B__inference_model_9_layer_call_and_return_conditional_losses_56065

inputs-
)fixed_adjacency_graph_convolution_2_56041-
)fixed_adjacency_graph_convolution_2_56043-
)fixed_adjacency_graph_convolution_2_56045
lstm_2_56051
lstm_2_56053
lstm_2_56055
dense_6_56059
dense_6_56061
identity??dense_6/StatefulPartitionedCall?;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimsinputs(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????F2
tf.expand_dims_4/ExpandDims?
reshape_17/PartitionedCallPartitionedCall$tf.expand_dims_4/ExpandDims:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_554412
reshape_17/PartitionedCall?
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_17/PartitionedCall:output:0)fixed_adjacency_graph_convolution_2_56041)fixed_adjacency_graph_convolution_2_56043)fixed_adjacency_graph_convolution_2_56045*
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
GPU 2J 8? *g
fbR`
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_555022=
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCallDfixed_adjacency_graph_convolution_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_18_layer_call_and_return_conditional_losses_555362
reshape_18/PartitionedCall?
permute_4/PartitionedCallPartitionedCall#reshape_18/PartitionedCall:output:0*
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
D__inference_permute_4_layer_call_and_return_conditional_losses_548102
permute_4/PartitionedCall?
reshape_19/PartitionedCallPartitionedCall"permute_4/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_19_layer_call_and_return_conditional_losses_555582
reshape_19/PartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall#reshape_19/PartitionedCall:output:0lstm_2_56051lstm_2_56053lstm_2_56055*
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_558632 
lstm_2/StatefulPartitionedCall?
dropout_7/PartitionedCallPartitionedCall'lstm_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_559102
dropout_7/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_6_56059dense_6_56061*
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
GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_559342!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall<^fixed_adjacency_graph_convolution_2/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????F::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2z
;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall;fixed_adjacency_graph_convolution_2/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
a
E__inference_reshape_19_layer_call_and_return_conditional_losses_55558

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
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_57057

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
?
while_cond_57820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_57820___redundant_placeholder03
/while_while_cond_57820___redundant_placeholder13
/while_while_cond_57820___redundant_placeholder23
/while_while_cond_57820___redundant_placeholder3
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
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54918

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
while_body_58290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_2_matmul_readvariableop_resource_08
4while_lstm_cell_2_matmul_1_readvariableop_resource_07
3while_lstm_cell_2_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_2_matmul_readvariableop_resource6
2while_lstm_cell_2_matmul_1_readvariableop_resource5
1while_lstm_cell_2_biasadd_readvariableop_resource??(while/lstm_cell_2/BiasAdd/ReadVariableOp?'while/lstm_cell_2/MatMul/ReadVariableOp?)while/lstm_cell_2/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	F?*
dtype02)
'while/lstm_cell_2/MatMul/ReadVariableOp?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul?
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/lstm_cell_2/MatMul_1/ReadVariableOp?
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/MatMul_1?
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add?
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_2/BiasAdd/ReadVariableOp?
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/BiasAddt
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_2/Const?
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_2/split/split_dim?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_2/split?
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid?
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_1?
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul?
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_1?
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/add_1?
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/Sigmoid_2?
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0while/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_2/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*
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
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 
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
(__inference_T-GCN-WX_layer_call_fn_56380
input_13
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
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8? *L
fGRE
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_563572
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
input_13
?
?
&__inference_lstm_2_layer_call_fn_58395
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
GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_554132
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
inputs/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_139
serving_default_input_13:0?????????F;
model_90
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
_tf_keras_network?U{"class_name": "Functional", "name": "T-GCN-WX", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_16", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["input_14", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_17", "inbound_nodes": [[["tf.expand_dims_4", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_2", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_18", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_18", "inbound_nodes": [[["fixed_adjacency_graph_convolution_2", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_4", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_4", "inbound_nodes": [[["reshape_18", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_19", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_19", "inbound_nodes": [[["permute_4", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["reshape_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "name": "model_9", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["model_9", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "T-GCN-WX", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}, "name": "input_13", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["input_13", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_16", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["input_14", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_17", "inbound_nodes": [[["tf.expand_dims_4", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_2", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_18", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_18", "inbound_nodes": [[["fixed_adjacency_graph_convolution_2", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_4", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_4", "inbound_nodes": [[["reshape_18", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_19", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_19", "inbound_nodes": [[["permute_4", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["reshape_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "name": "model_9", "inbound_nodes": [[["reshape_16", 0, 0, {}]]]}], "input_layers": [["input_13", 0, 0]], "output_layers": [["model_9", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_13", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12, 5]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
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
_tf_keras_network?<{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["input_14", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_17", "inbound_nodes": [[["tf.expand_dims_4", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_2", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_18", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_18", "inbound_nodes": [[["fixed_adjacency_graph_convolution_2", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_4", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_4", "inbound_nodes": [[["reshape_18", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_19", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_19", "inbound_nodes": [[["permute_4", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["reshape_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}, "name": "input_14", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["input_14", 0, 0, {"axis": -1}]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}, "name": "reshape_17", "inbound_nodes": [[["tf.expand_dims_4", 0, 0, {}]]]}, {"class_name": "FixedAdjacencyGraphConvolution", "config": {"name": "fixed_adjacency_graph_convolution_2", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "name": "fixed_adjacency_graph_convolution_2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_18", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}, "name": "reshape_18", "inbound_nodes": [[["fixed_adjacency_graph_convolution_2", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute_4", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "name": "permute_4", "inbound_nodes": [[["reshape_18", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_19", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}, "name": "reshape_19", "inbound_nodes": [[["permute_4", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["reshape_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_14", 0, 0]], "output_layers": [["dense_6", 0, 0]]}}}
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
 :2dense_7/kernel
:2dense_7/bias
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
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_14", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_14"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, 12]}}}
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
_tf_keras_layer?{"class_name": "FixedAdjacencyGraphConvolution", "name": "fixed_adjacency_graph_convolution_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_adjacency_graph_convolution_2", "trainable": true, "dtype": "float32", "units": 15, "use_bias": true, "activation": "linear", "kernel_initializer": null, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": null, "bias_regularizer": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 12]}}
?
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_18", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [70, -1, 1]}}}
?
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Permute", "name": "permute_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "permute_4", "trainable": true, "dtype": "float32", "dims": {"class_name": "__tuple__", "items": [2, 1, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_19", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 70]}}}
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

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 70]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 70]}}
?
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

2kernel
3bias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 70, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
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
<::2*fixed_adjacency_graph_convolution_2/kernel
::8F2(fixed_adjacency_graph_convolution_2/bias
,:*	F?2lstm_2/lstm_cell_2/kernel
7:5
??2#lstm_2/lstm_cell_2/recurrent_kernel
&:$?2lstm_2/lstm_cell_2/bias
!:	?F2dense_6/kernel
:F2dense_6/bias
5:3FF2%fixed_adjacency_graph_convolution_2/A
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
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
%:#2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
A:?21Adam/fixed_adjacency_graph_convolution_2/kernel/m
?:=F2/Adam/fixed_adjacency_graph_convolution_2/bias/m
1:/	F?2 Adam/lstm_2/lstm_cell_2/kernel/m
<::
??2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
+:)?2Adam/lstm_2/lstm_cell_2/bias/m
&:$	?F2Adam/dense_6/kernel/m
:F2Adam/dense_6/bias/m
%:#2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
A:?21Adam/fixed_adjacency_graph_convolution_2/kernel/v
?:=F2/Adam/fixed_adjacency_graph_convolution_2/bias/v
1:/	F?2 Adam/lstm_2/lstm_cell_2/kernel/v
<::
??2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
+:)?2Adam/lstm_2/lstm_cell_2/bias/v
&:$	?F2Adam/dense_6/kernel/v
:F2Adam/dense_6/bias/v
?2?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56703
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56273
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56977
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56245?
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
(__inference_T-GCN-WX_layer_call_fn_56380
(__inference_T-GCN-WX_layer_call_fn_57027
(__inference_T-GCN-WX_layer_call_fn_56327
(__inference_T-GCN-WX_layer_call_fn_57002?
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
 __inference__wrapped_model_54803?
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
input_13?????????F
?2?
B__inference_dense_7_layer_call_and_return_conditional_losses_57057?
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
'__inference_dense_7_layer_call_fn_57066?
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
D__inference_dropout_6_layer_call_and_return_conditional_losses_57083
D__inference_dropout_6_layer_call_and_return_conditional_losses_57078?
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
)__inference_dropout_6_layer_call_fn_57088
)__inference_dropout_6_layer_call_fn_57093?
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_57106?
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
*__inference_reshape_16_layer_call_fn_57111?
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
B__inference_model_9_layer_call_and_return_conditional_losses_57594
B__inference_model_9_layer_call_and_return_conditional_losses_57356
B__inference_model_9_layer_call_and_return_conditional_losses_55951
B__inference_model_9_layer_call_and_return_conditional_losses_55981?
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
'__inference_model_9_layer_call_fn_56033
'__inference_model_9_layer_call_fn_57636
'__inference_model_9_layer_call_fn_57615
'__inference_model_9_layer_call_fn_56084?
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
#__inference_signature_wrapper_56415input_13"?
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
E__inference_reshape_17_layer_call_and_return_conditional_losses_57649?
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
*__inference_reshape_17_layer_call_fn_57654?
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
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_57707?
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
C__inference_fixed_adjacency_graph_convolution_2_layer_call_fn_57718?
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_57732?
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
*__inference_reshape_18_layer_call_fn_57737?
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
D__inference_permute_4_layer_call_and_return_conditional_losses_54810?
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
)__inference_permute_4_layer_call_fn_54816?
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_57750?
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
*__inference_reshape_19_layer_call_fn_57755?
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_57904
A__inference_lstm_2_layer_call_and_return_conditional_losses_58224
A__inference_lstm_2_layer_call_and_return_conditional_losses_58053
A__inference_lstm_2_layer_call_and_return_conditional_losses_58373?
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
&__inference_lstm_2_layer_call_fn_58075
&__inference_lstm_2_layer_call_fn_58384
&__inference_lstm_2_layer_call_fn_58395
&__inference_lstm_2_layer_call_fn_58064?
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
D__inference_dropout_7_layer_call_and_return_conditional_losses_58407
D__inference_dropout_7_layer_call_and_return_conditional_losses_58412?
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
)__inference_dropout_7_layer_call_fn_58417
)__inference_dropout_7_layer_call_fn_58422?
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
B__inference_dense_6_layer_call_and_return_conditional_losses_58433?
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
'__inference_dense_6_layer_call_fn_58442?
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
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58504
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58473?
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
+__inference_lstm_cell_2_layer_call_fn_58521
+__inference_lstm_cell_2_layer_call_fn_58538?
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
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56245v
4-./0123A?>
7?4
*?'
input_13?????????F
p

 
? "%?"
?
0?????????F
? ?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56273v
4-./0123A?>
7?4
*?'
input_13?????????F
p 

 
? "%?"
?
0?????????F
? ?
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56703t
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
C__inference_T-GCN-WX_layer_call_and_return_conditional_losses_56977t
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
(__inference_T-GCN-WX_layer_call_fn_56327i
4-./0123A?>
7?4
*?'
input_13?????????F
p

 
? "??????????F?
(__inference_T-GCN-WX_layer_call_fn_56380i
4-./0123A?>
7?4
*?'
input_13?????????F
p 

 
? "??????????F?
(__inference_T-GCN-WX_layer_call_fn_57002g
4-./0123??<
5?2
(?%
inputs?????????F
p

 
? "??????????F?
(__inference_T-GCN-WX_layer_call_fn_57027g
4-./0123??<
5?2
(?%
inputs?????????F
p 

 
? "??????????F?
 __inference__wrapped_model_54803z
4-./01239?6
/?,
*?'
input_13?????????F
? "1?.
,
model_9!?
model_9?????????F?
B__inference_dense_6_layer_call_and_return_conditional_losses_58433]230?-
&?#
!?
inputs??????????
? "%?"
?
0?????????F
? {
'__inference_dense_6_layer_call_fn_58442P230?-
&?#
!?
inputs??????????
? "??????????F?
B__inference_dense_7_layer_call_and_return_conditional_losses_57057l7?4
-?*
(?%
inputs?????????F
? "-?*
#? 
0?????????F
? ?
'__inference_dense_7_layer_call_fn_57066_7?4
-?*
(?%
inputs?????????F
? " ??????????F?
D__inference_dropout_6_layer_call_and_return_conditional_losses_57078l;?8
1?.
(?%
inputs?????????F
p
? "-?*
#? 
0?????????F
? ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_57083l;?8
1?.
(?%
inputs?????????F
p 
? "-?*
#? 
0?????????F
? ?
)__inference_dropout_6_layer_call_fn_57088_;?8
1?.
(?%
inputs?????????F
p
? " ??????????F?
)__inference_dropout_6_layer_call_fn_57093_;?8
1?.
(?%
inputs?????????F
p 
? " ??????????F?
D__inference_dropout_7_layer_call_and_return_conditional_losses_58407^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_58412^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_7_layer_call_fn_58417Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_7_layer_call_fn_58422Q4?1
*?'
!?
inputs??????????
p 
? "????????????
^__inference_fixed_adjacency_graph_convolution_2_layer_call_and_return_conditional_losses_57707g4-.5?2
+?(
&?#
features?????????F
? ")?&
?
0?????????F
? ?
C__inference_fixed_adjacency_graph_convolution_2_layer_call_fn_57718Z4-.5?2
+?(
&?#
features?????????F
? "??????????F?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57904n/01??<
5?2
$?!
inputs?????????F

 
p

 
? "&?#
?
0??????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58053n/01??<
5?2
$?!
inputs?????????F

 
p 

 
? "&?#
?
0??????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58224~/01O?L
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_58373~/01O?L
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
&__inference_lstm_2_layer_call_fn_58064a/01??<
5?2
$?!
inputs?????????F

 
p

 
? "????????????
&__inference_lstm_2_layer_call_fn_58075a/01??<
5?2
$?!
inputs?????????F

 
p 

 
? "????????????
&__inference_lstm_2_layer_call_fn_58384q/01O?L
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
&__inference_lstm_2_layer_call_fn_58395q/01O?L
E?B
4?1
/?,
inputs/0??????????????????F

 
p 

 
? "????????????
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58473?/01??
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
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_58504?/01??
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
+__inference_lstm_cell_2_layer_call_fn_58521?/01??
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
+__inference_lstm_cell_2_layer_call_fn_58538?/01??
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
B__inference_model_9_layer_call_and_return_conditional_losses_55951p4-./0123=?:
3?0
&?#
input_14?????????F
p

 
? "%?"
?
0?????????F
? ?
B__inference_model_9_layer_call_and_return_conditional_losses_55981p4-./0123=?:
3?0
&?#
input_14?????????F
p 

 
? "%?"
?
0?????????F
? ?
B__inference_model_9_layer_call_and_return_conditional_losses_57356n4-./0123;?8
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
B__inference_model_9_layer_call_and_return_conditional_losses_57594n4-./0123;?8
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
'__inference_model_9_layer_call_fn_56033c4-./0123=?:
3?0
&?#
input_14?????????F
p

 
? "??????????F?
'__inference_model_9_layer_call_fn_56084c4-./0123=?:
3?0
&?#
input_14?????????F
p 

 
? "??????????F?
'__inference_model_9_layer_call_fn_57615a4-./0123;?8
1?.
$?!
inputs?????????F
p

 
? "??????????F?
'__inference_model_9_layer_call_fn_57636a4-./0123;?8
1?.
$?!
inputs?????????F
p 

 
? "??????????F?
D__inference_permute_4_layer_call_and_return_conditional_losses_54810?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_permute_4_layer_call_fn_54816?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_reshape_16_layer_call_and_return_conditional_losses_57106d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
*__inference_reshape_16_layer_call_fn_57111W7?4
-?*
(?%
inputs?????????F
? "??????????F?
E__inference_reshape_17_layer_call_and_return_conditional_losses_57649d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
*__inference_reshape_17_layer_call_fn_57654W7?4
-?*
(?%
inputs?????????F
? "??????????F?
E__inference_reshape_18_layer_call_and_return_conditional_losses_57732d3?0
)?&
$?!
inputs?????????F
? "-?*
#? 
0?????????F
? ?
*__inference_reshape_18_layer_call_fn_57737W3?0
)?&
$?!
inputs?????????F
? " ??????????F?
E__inference_reshape_19_layer_call_and_return_conditional_losses_57750d7?4
-?*
(?%
inputs?????????F
? ")?&
?
0?????????F
? ?
*__inference_reshape_19_layer_call_fn_57755W7?4
-?*
(?%
inputs?????????F
? "??????????F?
#__inference_signature_wrapper_56415?
4-./0123E?B
? 
;?8
6
input_13*?'
input_13?????????F"1?.
,
model_9!?
model_9?????????F