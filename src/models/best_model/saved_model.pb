??
??
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:d*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:d*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
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
}
dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*!
shared_namedense_4/kernel/m
v
$dense_4/kernel/m/Read/ReadVariableOpReadVariableOpdense_4/kernel/m*
_output_shapes
:	?d*
dtype0
t
dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias/m
m
"dense_4/bias/m/Read/ReadVariableOpReadVariableOpdense_4/bias/m*
_output_shapes
:d*
dtype0
|
dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_5/kernel/m
u
$dense_5/kernel/m/Read/ReadVariableOpReadVariableOpdense_5/kernel/m*
_output_shapes

:d*
dtype0
t
dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/m
m
"dense_5/bias/m/Read/ReadVariableOpReadVariableOpdense_5/bias/m*
_output_shapes
:*
dtype0
~
dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_2/kernel/m
w
$dense_2/kernel/m/Read/ReadVariableOpReadVariableOpdense_2/kernel/m* 
_output_shapes
:
??*
dtype0
u
dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias/m
n
"dense_2/bias/m/Read/ReadVariableOpReadVariableOpdense_2/bias/m*
_output_shapes	
:?*
dtype0
}
dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_3/kernel/m
v
$dense_3/kernel/m/Read/ReadVariableOpReadVariableOpdense_3/kernel/m*
_output_shapes
:	?*
dtype0
t
dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias/m
m
"dense_3/bias/m/Read/ReadVariableOpReadVariableOpdense_3/bias/m*
_output_shapes
:*
dtype0
}
dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*!
shared_namedense_4/kernel/v
v
$dense_4/kernel/v/Read/ReadVariableOpReadVariableOpdense_4/kernel/v*
_output_shapes
:	?d*
dtype0
t
dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias/v
m
"dense_4/bias/v/Read/ReadVariableOpReadVariableOpdense_4/bias/v*
_output_shapes
:d*
dtype0
|
dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_5/kernel/v
u
$dense_5/kernel/v/Read/ReadVariableOpReadVariableOpdense_5/kernel/v*
_output_shapes

:d*
dtype0
t
dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/v
m
"dense_5/bias/v/Read/ReadVariableOpReadVariableOpdense_5/bias/v*
_output_shapes
:*
dtype0
~
dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_2/kernel/v
w
$dense_2/kernel/v/Read/ReadVariableOpReadVariableOpdense_2/kernel/v* 
_output_shapes
:
??*
dtype0
u
dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias/v
n
"dense_2/bias/v/Read/ReadVariableOpReadVariableOpdense_2/bias/v*
_output_shapes	
:?*
dtype0
}
dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_3/kernel/v
v
$dense_3/kernel/v/Read/ReadVariableOpReadVariableOpdense_3/kernel/v*
_output_shapes
:	?*
dtype0
t
dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias/v
m
"dense_3/bias/v/Read/ReadVariableOpReadVariableOpdense_3/bias/v*
_output_shapes
:*
dtype0
?
dense_2/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namedense_2/kernel/m_1
{
&dense_2/kernel/m_1/Read/ReadVariableOpReadVariableOpdense_2/kernel/m_1* 
_output_shapes
:
??*
dtype0
y
dense_2/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedense_2/bias/m_1
r
$dense_2/bias/m_1/Read/ReadVariableOpReadVariableOpdense_2/bias/m_1*
_output_shapes	
:?*
dtype0
?
dense_3/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namedense_3/kernel/m_1
z
&dense_3/kernel/m_1/Read/ReadVariableOpReadVariableOpdense_3/kernel/m_1*
_output_shapes
:	?*
dtype0
x
dense_3/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_3/bias/m_1
q
$dense_3/bias/m_1/Read/ReadVariableOpReadVariableOpdense_3/bias/m_1*
_output_shapes
:*
dtype0
?
dense_2/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namedense_2/kernel/v_1
{
&dense_2/kernel/v_1/Read/ReadVariableOpReadVariableOpdense_2/kernel/v_1* 
_output_shapes
:
??*
dtype0
y
dense_2/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedense_2/bias/v_1
r
$dense_2/bias/v_1/Read/ReadVariableOpReadVariableOpdense_2/bias/v_1*
_output_shapes	
:?*
dtype0
?
dense_3/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namedense_3/kernel/v_1
z
&dense_3/kernel/v_1/Read/ReadVariableOpReadVariableOpdense_3/kernel/v_1*
_output_shapes
:	?*
dtype0
x
dense_3/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_3/bias/v_1
q
$dense_3/bias/v_1/Read/ReadVariableOpReadVariableOpdense_3/bias/v_1*
_output_shapes
:*
dtype0

NoOpNoOp
?8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-1
layer-17
layer_with_weights-2
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 

	keras_api

	keras_api

	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$	keras_api

%	keras_api

&	keras_api

'	keras_api

(	keras_api

)	keras_api

*	keras_api

+	keras_api

,	keras_api

-	keras_api

.	keras_api

/	keras_api
h

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?0mq1mr6ms7mt<mu=mv>mw?mx0vy1vz6v{7v|<v}=v~>v?v?
 
8
<0
=1
>2
?3
04
15
66
77
8
<0
=1
>2
?3
04
15
66
77
?
@non_trainable_variables
regularization_losses

Alayers
Blayer_regularization_losses
Cmetrics
Dlayer_metrics
	variables
trainable_variables
 
 
 
 
h

<kernel
=bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
h

>kernel
?bias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
P<m?=m?>m??m?<v?=v?>v??v?
 

<0
=1
>2
?3

<0
=1
>2
?3
?
Mnon_trainable_variables
 regularization_losses

Nlayers
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
!	variables
"trainable_variables
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
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
?
Rnon_trainable_variables
2regularization_losses

Slayers
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
3	variables
4trainable_variables
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
Wnon_trainable_variables
8regularization_losses

Xlayers
Ylayer_regularization_losses
Zmetrics
[layer_metrics
9	variables
:trainable_variables
JH
VARIABLE_VALUEdense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_3/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
?
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
10
11
12
13
14
15
16
17
18
 

\0
]1
 
 

<0
=1

<0
=1
?
^non_trainable_variables
Eregularization_losses

_layers
`layer_regularization_losses
ametrics
blayer_metrics
F	variables
Gtrainable_variables
 

>0
?1

>0
?1
?
cnon_trainable_variables
Iregularization_losses

dlayers
elayer_regularization_losses
fmetrics
glayer_metrics
J	variables
Ktrainable_variables
 

0
1
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
	htotal
	icount
j	variables
k	keras_api
D
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api
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

h0
i1

j	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

o	variables
xv
VARIABLE_VALUEdense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEdense_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEdense_2/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEdense_3/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEdense_3/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEdense_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEdense_2/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEdense_3/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEdense_3/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEdense_2/kernel/m_1Wvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEdense_2/bias/m_1Wvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEdense_3/kernel/m_1Wvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEdense_3/bias/m_1Wvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEdense_2/kernel/v_1Wvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEdense_2/bias/v_1Wvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEdense_3/kernel/v_1Wvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEdense_3/bias/v_1Wvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*,
_output_shapes
:?????????
?*
dtype0*!
shape:?????????
?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_9654
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$dense_4/kernel/m/Read/ReadVariableOp"dense_4/bias/m/Read/ReadVariableOp$dense_5/kernel/m/Read/ReadVariableOp"dense_5/bias/m/Read/ReadVariableOp$dense_2/kernel/m/Read/ReadVariableOp"dense_2/bias/m/Read/ReadVariableOp$dense_3/kernel/m/Read/ReadVariableOp"dense_3/bias/m/Read/ReadVariableOp$dense_4/kernel/v/Read/ReadVariableOp"dense_4/bias/v/Read/ReadVariableOp$dense_5/kernel/v/Read/ReadVariableOp"dense_5/bias/v/Read/ReadVariableOp$dense_2/kernel/v/Read/ReadVariableOp"dense_2/bias/v/Read/ReadVariableOp$dense_3/kernel/v/Read/ReadVariableOp"dense_3/bias/v/Read/ReadVariableOp&dense_2/kernel/m_1/Read/ReadVariableOp$dense_2/bias/m_1/Read/ReadVariableOp&dense_3/kernel/m_1/Read/ReadVariableOp$dense_3/bias/m_1/Read/ReadVariableOp&dense_2/kernel/v_1/Read/ReadVariableOp$dense_2/bias/v_1/Read/ReadVariableOp&dense_3/kernel/v_1/Read/ReadVariableOp$dense_3/bias/v_1/Read/ReadVariableOpConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_10274
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotalcounttotal_1count_1dense_4/kernel/mdense_4/bias/mdense_5/kernel/mdense_5/bias/mdense_2/kernel/mdense_2/bias/mdense_3/kernel/mdense_3/bias/mdense_4/kernel/vdense_4/bias/vdense_5/kernel/vdense_5/bias/vdense_2/kernel/vdense_2/bias/vdense_3/kernel/vdense_3/bias/vdense_2/kernel/m_1dense_2/bias/m_1dense_3/kernel/m_1dense_3/bias/m_1dense_2/kernel/v_1dense_2/bias/v_1dense_3/kernel/v_1dense_3/bias/v_1*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_10392??
?
?
'__inference_dense_5_layer_call_fn_10104

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_91922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?L
?
__inference__traced_save_10274
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop/
+savev2_dense_4_kernel_m_read_readvariableop-
)savev2_dense_4_bias_m_read_readvariableop/
+savev2_dense_5_kernel_m_read_readvariableop-
)savev2_dense_5_bias_m_read_readvariableop/
+savev2_dense_2_kernel_m_read_readvariableop-
)savev2_dense_2_bias_m_read_readvariableop/
+savev2_dense_3_kernel_m_read_readvariableop-
)savev2_dense_3_bias_m_read_readvariableop/
+savev2_dense_4_kernel_v_read_readvariableop-
)savev2_dense_4_bias_v_read_readvariableop/
+savev2_dense_5_kernel_v_read_readvariableop-
)savev2_dense_5_bias_v_read_readvariableop/
+savev2_dense_2_kernel_v_read_readvariableop-
)savev2_dense_2_bias_v_read_readvariableop/
+savev2_dense_3_kernel_v_read_readvariableop-
)savev2_dense_3_bias_v_read_readvariableop1
-savev2_dense_2_kernel_m_1_read_readvariableop/
+savev2_dense_2_bias_m_1_read_readvariableop1
-savev2_dense_3_kernel_m_1_read_readvariableop/
+savev2_dense_3_bias_m_1_read_readvariableop1
-savev2_dense_2_kernel_v_1_read_readvariableop/
+savev2_dense_2_bias_v_1_read_readvariableop1
-savev2_dense_3_kernel_v_1_read_readvariableop/
+savev2_dense_3_bias_v_1_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_dense_4_kernel_m_read_readvariableop)savev2_dense_4_bias_m_read_readvariableop+savev2_dense_5_kernel_m_read_readvariableop)savev2_dense_5_bias_m_read_readvariableop+savev2_dense_2_kernel_m_read_readvariableop)savev2_dense_2_bias_m_read_readvariableop+savev2_dense_3_kernel_m_read_readvariableop)savev2_dense_3_bias_m_read_readvariableop+savev2_dense_4_kernel_v_read_readvariableop)savev2_dense_4_bias_v_read_readvariableop+savev2_dense_5_kernel_v_read_readvariableop)savev2_dense_5_bias_v_read_readvariableop+savev2_dense_2_kernel_v_read_readvariableop)savev2_dense_2_bias_v_read_readvariableop+savev2_dense_3_kernel_v_read_readvariableop)savev2_dense_3_bias_v_read_readvariableop-savev2_dense_2_kernel_m_1_read_readvariableop+savev2_dense_2_bias_m_1_read_readvariableop-savev2_dense_3_kernel_m_1_read_readvariableop+savev2_dense_3_bias_m_1_read_readvariableop-savev2_dense_2_kernel_v_1_read_readvariableop+savev2_dense_2_bias_v_1_read_readvariableop-savev2_dense_3_kernel_v_1_read_readvariableop+savev2_dense_3_bias_v_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%2
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
?: :	?d:d:d::
??:?:	?:: : : : :	?d:d:d::
??:?:	?::	?d:d:d::
??:?:	?::
??:?:	?::
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:  

_output_shapes
::&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:%#!

_output_shapes
:	?: $

_output_shapes
::%

_output_shapes
: 
?y
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9521
input_2#
statsmodel_9424:
??
statsmodel_9426:	?"
statsmodel_9428:	?
statsmodel_9430:
dense_4_9510:	?d
dense_4_9512:d
dense_5_9515:d
dense_5_9517:
identity??"StatsModel/StatefulPartitionedCall?$StatsModel/StatefulPartitionedCall_1?$StatsModel/StatefulPartitionedCall_2?$StatsModel/StatefulPartitionedCall_3?$StatsModel/StatefulPartitionedCall_4?$StatsModel/StatefulPartitionedCall_5?$StatsModel/StatefulPartitionedCall_6?$StatsModel/StatefulPartitionedCall_7?$StatsModel/StatefulPartitionedCall_8?$StatsModel/StatefulPartitionedCall_9?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
tf.unstack_1/unstackUnpackinput_2*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
"StatsModel/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_1/unstack:output:5statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492$
"StatsModel/StatefulPartitionedCall?
$StatsModel/StatefulPartitionedCall_1StatefulPartitionedCalltf.unstack_1/unstack:output:6statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_1?
$StatsModel/StatefulPartitionedCall_2StatefulPartitionedCalltf.unstack_1/unstack:output:7statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_2?
$StatsModel/StatefulPartitionedCall_3StatefulPartitionedCalltf.unstack_1/unstack:output:8statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_3?
$StatsModel/StatefulPartitionedCall_4StatefulPartitionedCalltf.unstack_1/unstack:output:9statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_4?
$StatsModel/StatefulPartitionedCall_5StatefulPartitionedCalltf.unstack_1/unstack:output:0statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_5?
$StatsModel/StatefulPartitionedCall_6StatefulPartitionedCalltf.unstack_1/unstack:output:1statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_6?
$StatsModel/StatefulPartitionedCall_7StatefulPartitionedCalltf.unstack_1/unstack:output:2statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_7?
$StatsModel/StatefulPartitionedCall_8StatefulPartitionedCalltf.unstack_1/unstack:output:3statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_8?
$StatsModel/StatefulPartitionedCall_9StatefulPartitionedCalltf.unstack_1/unstack:output:4statsmodel_9424statsmodel_9426statsmodel_9428statsmodel_9430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_4_9510dense_4_9512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_91752!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9515dense_5_9517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_91922!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0#^StatsModel/StatefulPartitionedCall%^StatsModel/StatefulPartitionedCall_1%^StatsModel/StatefulPartitionedCall_2%^StatsModel/StatefulPartitionedCall_3%^StatsModel/StatefulPartitionedCall_4%^StatsModel/StatefulPartitionedCall_5%^StatsModel/StatefulPartitionedCall_6%^StatsModel/StatefulPartitionedCall_7%^StatsModel/StatefulPartitionedCall_8%^StatsModel/StatefulPartitionedCall_9 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2H
"StatsModel/StatefulPartitionedCall"StatsModel/StatefulPartitionedCall2L
$StatsModel/StatefulPartitionedCall_1$StatsModel/StatefulPartitionedCall_12L
$StatsModel/StatefulPartitionedCall_2$StatsModel/StatefulPartitionedCall_22L
$StatsModel/StatefulPartitionedCall_3$StatsModel/StatefulPartitionedCall_32L
$StatsModel/StatefulPartitionedCall_4$StatsModel/StatefulPartitionedCall_42L
$StatsModel/StatefulPartitionedCall_5$StatsModel/StatefulPartitionedCall_52L
$StatsModel/StatefulPartitionedCall_6$StatsModel/StatefulPartitionedCall_62L
$StatsModel/StatefulPartitionedCall_7$StatsModel/StatefulPartitionedCall_72L
$StatsModel/StatefulPartitionedCall_8$StatsModel/StatefulPartitionedCall_82L
$StatsModel/StatefulPartitionedCall_9$StatsModel/StatefulPartitionedCall_92B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?y
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9371

inputs#
statsmodel_9274:
??
statsmodel_9276:	?"
statsmodel_9278:	?
statsmodel_9280:
dense_4_9360:	?d
dense_4_9362:d
dense_5_9365:d
dense_5_9367:
identity??"StatsModel/StatefulPartitionedCall?$StatsModel/StatefulPartitionedCall_1?$StatsModel/StatefulPartitionedCall_2?$StatsModel/StatefulPartitionedCall_3?$StatsModel/StatefulPartitionedCall_4?$StatsModel/StatefulPartitionedCall_5?$StatsModel/StatefulPartitionedCall_6?$StatsModel/StatefulPartitionedCall_7?$StatsModel/StatefulPartitionedCall_8?$StatsModel/StatefulPartitionedCall_9?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
tf.unstack_1/unstackUnpackinputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
"StatsModel/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_1/unstack:output:5statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092$
"StatsModel/StatefulPartitionedCall?
$StatsModel/StatefulPartitionedCall_1StatefulPartitionedCalltf.unstack_1/unstack:output:6statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_1?
$StatsModel/StatefulPartitionedCall_2StatefulPartitionedCalltf.unstack_1/unstack:output:7statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_2?
$StatsModel/StatefulPartitionedCall_3StatefulPartitionedCalltf.unstack_1/unstack:output:8statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_3?
$StatsModel/StatefulPartitionedCall_4StatefulPartitionedCalltf.unstack_1/unstack:output:9statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_4?
$StatsModel/StatefulPartitionedCall_5StatefulPartitionedCalltf.unstack_1/unstack:output:0statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_5?
$StatsModel/StatefulPartitionedCall_6StatefulPartitionedCalltf.unstack_1/unstack:output:1statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_6?
$StatsModel/StatefulPartitionedCall_7StatefulPartitionedCalltf.unstack_1/unstack:output:2statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_7?
$StatsModel/StatefulPartitionedCall_8StatefulPartitionedCalltf.unstack_1/unstack:output:3statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_8?
$StatsModel/StatefulPartitionedCall_9StatefulPartitionedCalltf.unstack_1/unstack:output:4statsmodel_9274statsmodel_9276statsmodel_9278statsmodel_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_4_9360dense_4_9362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_91752!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9365dense_5_9367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_91922!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0#^StatsModel/StatefulPartitionedCall%^StatsModel/StatefulPartitionedCall_1%^StatsModel/StatefulPartitionedCall_2%^StatsModel/StatefulPartitionedCall_3%^StatsModel/StatefulPartitionedCall_4%^StatsModel/StatefulPartitionedCall_5%^StatsModel/StatefulPartitionedCall_6%^StatsModel/StatefulPartitionedCall_7%^StatsModel/StatefulPartitionedCall_8%^StatsModel/StatefulPartitionedCall_9 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2H
"StatsModel/StatefulPartitionedCall"StatsModel/StatefulPartitionedCall2L
$StatsModel/StatefulPartitionedCall_1$StatsModel/StatefulPartitionedCall_12L
$StatsModel/StatefulPartitionedCall_2$StatsModel/StatefulPartitionedCall_22L
$StatsModel/StatefulPartitionedCall_3$StatsModel/StatefulPartitionedCall_32L
$StatsModel/StatefulPartitionedCall_4$StatsModel/StatefulPartitionedCall_42L
$StatsModel/StatefulPartitionedCall_5$StatsModel/StatefulPartitionedCall_52L
$StatsModel/StatefulPartitionedCall_6$StatsModel/StatefulPartitionedCall_62L
$StatsModel/StatefulPartitionedCall_7$StatsModel/StatefulPartitionedCall_72L
$StatsModel/StatefulPartitionedCall_8$StatsModel/StatefulPartitionedCall_82L
$StatsModel/StatefulPartitionedCall_9$StatsModel/StatefulPartitionedCall_92B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
)__inference_StatsModel_layer_call_fn_8960
dense_2_input
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_2_input
?
?
'__inference_dense_3_layer_call_fn_10143

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_89422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_10124

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_89262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_10095

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
D__inference_StatsModel_layer_call_and_return_conditional_losses_9047
dense_2_input 
dense_2_9036:
??
dense_2_9038:	?
dense_3_9041:	?
dense_3_9043:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_9036dense_2_9038*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_89262!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_9041dense_3_9043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_89422!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_2_input
̖
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9808

inputsE
1statsmodel_dense_2_matmul_readvariableop_resource:
??A
2statsmodel_dense_2_biasadd_readvariableop_resource:	?D
1statsmodel_dense_3_matmul_readvariableop_resource:	?@
2statsmodel_dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?d5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:d5
'dense_5_biasadd_readvariableop_resource:
identity??)StatsModel/dense_2/BiasAdd/ReadVariableOp?+StatsModel/dense_2/BiasAdd_1/ReadVariableOp?+StatsModel/dense_2/BiasAdd_2/ReadVariableOp?+StatsModel/dense_2/BiasAdd_3/ReadVariableOp?+StatsModel/dense_2/BiasAdd_4/ReadVariableOp?+StatsModel/dense_2/BiasAdd_5/ReadVariableOp?+StatsModel/dense_2/BiasAdd_6/ReadVariableOp?+StatsModel/dense_2/BiasAdd_7/ReadVariableOp?+StatsModel/dense_2/BiasAdd_8/ReadVariableOp?+StatsModel/dense_2/BiasAdd_9/ReadVariableOp?(StatsModel/dense_2/MatMul/ReadVariableOp?*StatsModel/dense_2/MatMul_1/ReadVariableOp?*StatsModel/dense_2/MatMul_2/ReadVariableOp?*StatsModel/dense_2/MatMul_3/ReadVariableOp?*StatsModel/dense_2/MatMul_4/ReadVariableOp?*StatsModel/dense_2/MatMul_5/ReadVariableOp?*StatsModel/dense_2/MatMul_6/ReadVariableOp?*StatsModel/dense_2/MatMul_7/ReadVariableOp?*StatsModel/dense_2/MatMul_8/ReadVariableOp?*StatsModel/dense_2/MatMul_9/ReadVariableOp?)StatsModel/dense_3/BiasAdd/ReadVariableOp?+StatsModel/dense_3/BiasAdd_1/ReadVariableOp?+StatsModel/dense_3/BiasAdd_2/ReadVariableOp?+StatsModel/dense_3/BiasAdd_3/ReadVariableOp?+StatsModel/dense_3/BiasAdd_4/ReadVariableOp?+StatsModel/dense_3/BiasAdd_5/ReadVariableOp?+StatsModel/dense_3/BiasAdd_6/ReadVariableOp?+StatsModel/dense_3/BiasAdd_7/ReadVariableOp?+StatsModel/dense_3/BiasAdd_8/ReadVariableOp?+StatsModel/dense_3/BiasAdd_9/ReadVariableOp?(StatsModel/dense_3/MatMul/ReadVariableOp?*StatsModel/dense_3/MatMul_1/ReadVariableOp?*StatsModel/dense_3/MatMul_2/ReadVariableOp?*StatsModel/dense_3/MatMul_3/ReadVariableOp?*StatsModel/dense_3/MatMul_4/ReadVariableOp?*StatsModel/dense_3/MatMul_5/ReadVariableOp?*StatsModel/dense_3/MatMul_6/ReadVariableOp?*StatsModel/dense_3/MatMul_7/ReadVariableOp?*StatsModel/dense_3/MatMul_8/ReadVariableOp?*StatsModel/dense_3/MatMul_9/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
tf.unstack_1/unstackUnpackinputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
(StatsModel/dense_2/MatMul/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(StatsModel/dense_2/MatMul/ReadVariableOp?
StatsModel/dense_2/MatMulMatMultf.unstack_1/unstack:output:50StatsModel/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul?
)StatsModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)StatsModel/dense_2/BiasAdd/ReadVariableOp?
StatsModel/dense_2/BiasAddBiasAdd#StatsModel/dense_2/MatMul:product:01StatsModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd?
StatsModel/dense_2/ReluRelu#StatsModel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu?
(StatsModel/dense_3/MatMul/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(StatsModel/dense_3/MatMul/ReadVariableOp?
StatsModel/dense_3/MatMulMatMul%StatsModel/dense_2/Relu:activations:00StatsModel/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul?
)StatsModel/dense_3/BiasAdd/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)StatsModel/dense_3/BiasAdd/ReadVariableOp?
StatsModel/dense_3/BiasAddBiasAdd#StatsModel/dense_3/MatMul:product:01StatsModel/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd?
*StatsModel/dense_2/MatMul_1/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_1/ReadVariableOp?
StatsModel/dense_2/MatMul_1MatMultf.unstack_1/unstack:output:62StatsModel/dense_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_1?
+StatsModel/dense_2/BiasAdd_1/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_1/ReadVariableOp?
StatsModel/dense_2/BiasAdd_1BiasAdd%StatsModel/dense_2/MatMul_1:product:03StatsModel/dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_1?
StatsModel/dense_2/Relu_1Relu%StatsModel/dense_2/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_1?
*StatsModel/dense_3/MatMul_1/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_1/ReadVariableOp?
StatsModel/dense_3/MatMul_1MatMul'StatsModel/dense_2/Relu_1:activations:02StatsModel/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_1?
+StatsModel/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_1/ReadVariableOp?
StatsModel/dense_3/BiasAdd_1BiasAdd%StatsModel/dense_3/MatMul_1:product:03StatsModel/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_1?
*StatsModel/dense_2/MatMul_2/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_2/ReadVariableOp?
StatsModel/dense_2/MatMul_2MatMultf.unstack_1/unstack:output:72StatsModel/dense_2/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_2?
+StatsModel/dense_2/BiasAdd_2/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_2/ReadVariableOp?
StatsModel/dense_2/BiasAdd_2BiasAdd%StatsModel/dense_2/MatMul_2:product:03StatsModel/dense_2/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_2?
StatsModel/dense_2/Relu_2Relu%StatsModel/dense_2/BiasAdd_2:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_2?
*StatsModel/dense_3/MatMul_2/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_2/ReadVariableOp?
StatsModel/dense_3/MatMul_2MatMul'StatsModel/dense_2/Relu_2:activations:02StatsModel/dense_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_2?
+StatsModel/dense_3/BiasAdd_2/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_2/ReadVariableOp?
StatsModel/dense_3/BiasAdd_2BiasAdd%StatsModel/dense_3/MatMul_2:product:03StatsModel/dense_3/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_2?
*StatsModel/dense_2/MatMul_3/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_3/ReadVariableOp?
StatsModel/dense_2/MatMul_3MatMultf.unstack_1/unstack:output:82StatsModel/dense_2/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_3?
+StatsModel/dense_2/BiasAdd_3/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_3/ReadVariableOp?
StatsModel/dense_2/BiasAdd_3BiasAdd%StatsModel/dense_2/MatMul_3:product:03StatsModel/dense_2/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_3?
StatsModel/dense_2/Relu_3Relu%StatsModel/dense_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_3?
*StatsModel/dense_3/MatMul_3/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_3/ReadVariableOp?
StatsModel/dense_3/MatMul_3MatMul'StatsModel/dense_2/Relu_3:activations:02StatsModel/dense_3/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_3?
+StatsModel/dense_3/BiasAdd_3/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_3/ReadVariableOp?
StatsModel/dense_3/BiasAdd_3BiasAdd%StatsModel/dense_3/MatMul_3:product:03StatsModel/dense_3/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_3?
*StatsModel/dense_2/MatMul_4/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_4/ReadVariableOp?
StatsModel/dense_2/MatMul_4MatMultf.unstack_1/unstack:output:92StatsModel/dense_2/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_4?
+StatsModel/dense_2/BiasAdd_4/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_4/ReadVariableOp?
StatsModel/dense_2/BiasAdd_4BiasAdd%StatsModel/dense_2/MatMul_4:product:03StatsModel/dense_2/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_4?
StatsModel/dense_2/Relu_4Relu%StatsModel/dense_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_4?
*StatsModel/dense_3/MatMul_4/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_4/ReadVariableOp?
StatsModel/dense_3/MatMul_4MatMul'StatsModel/dense_2/Relu_4:activations:02StatsModel/dense_3/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_4?
+StatsModel/dense_3/BiasAdd_4/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_4/ReadVariableOp?
StatsModel/dense_3/BiasAdd_4BiasAdd%StatsModel/dense_3/MatMul_4:product:03StatsModel/dense_3/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_4?
*StatsModel/dense_2/MatMul_5/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_5/ReadVariableOp?
StatsModel/dense_2/MatMul_5MatMultf.unstack_1/unstack:output:02StatsModel/dense_2/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_5?
+StatsModel/dense_2/BiasAdd_5/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_5/ReadVariableOp?
StatsModel/dense_2/BiasAdd_5BiasAdd%StatsModel/dense_2/MatMul_5:product:03StatsModel/dense_2/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_5?
StatsModel/dense_2/Relu_5Relu%StatsModel/dense_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_5?
*StatsModel/dense_3/MatMul_5/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_5/ReadVariableOp?
StatsModel/dense_3/MatMul_5MatMul'StatsModel/dense_2/Relu_5:activations:02StatsModel/dense_3/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_5?
+StatsModel/dense_3/BiasAdd_5/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_5/ReadVariableOp?
StatsModel/dense_3/BiasAdd_5BiasAdd%StatsModel/dense_3/MatMul_5:product:03StatsModel/dense_3/BiasAdd_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_5?
*StatsModel/dense_2/MatMul_6/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_6/ReadVariableOp?
StatsModel/dense_2/MatMul_6MatMultf.unstack_1/unstack:output:12StatsModel/dense_2/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_6?
+StatsModel/dense_2/BiasAdd_6/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_6/ReadVariableOp?
StatsModel/dense_2/BiasAdd_6BiasAdd%StatsModel/dense_2/MatMul_6:product:03StatsModel/dense_2/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_6?
StatsModel/dense_2/Relu_6Relu%StatsModel/dense_2/BiasAdd_6:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_6?
*StatsModel/dense_3/MatMul_6/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_6/ReadVariableOp?
StatsModel/dense_3/MatMul_6MatMul'StatsModel/dense_2/Relu_6:activations:02StatsModel/dense_3/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_6?
+StatsModel/dense_3/BiasAdd_6/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_6/ReadVariableOp?
StatsModel/dense_3/BiasAdd_6BiasAdd%StatsModel/dense_3/MatMul_6:product:03StatsModel/dense_3/BiasAdd_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_6?
*StatsModel/dense_2/MatMul_7/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_7/ReadVariableOp?
StatsModel/dense_2/MatMul_7MatMultf.unstack_1/unstack:output:22StatsModel/dense_2/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_7?
+StatsModel/dense_2/BiasAdd_7/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_7/ReadVariableOp?
StatsModel/dense_2/BiasAdd_7BiasAdd%StatsModel/dense_2/MatMul_7:product:03StatsModel/dense_2/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_7?
StatsModel/dense_2/Relu_7Relu%StatsModel/dense_2/BiasAdd_7:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_7?
*StatsModel/dense_3/MatMul_7/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_7/ReadVariableOp?
StatsModel/dense_3/MatMul_7MatMul'StatsModel/dense_2/Relu_7:activations:02StatsModel/dense_3/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_7?
+StatsModel/dense_3/BiasAdd_7/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_7/ReadVariableOp?
StatsModel/dense_3/BiasAdd_7BiasAdd%StatsModel/dense_3/MatMul_7:product:03StatsModel/dense_3/BiasAdd_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_7?
*StatsModel/dense_2/MatMul_8/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_8/ReadVariableOp?
StatsModel/dense_2/MatMul_8MatMultf.unstack_1/unstack:output:32StatsModel/dense_2/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_8?
+StatsModel/dense_2/BiasAdd_8/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_8/ReadVariableOp?
StatsModel/dense_2/BiasAdd_8BiasAdd%StatsModel/dense_2/MatMul_8:product:03StatsModel/dense_2/BiasAdd_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_8?
StatsModel/dense_2/Relu_8Relu%StatsModel/dense_2/BiasAdd_8:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_8?
*StatsModel/dense_3/MatMul_8/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_8/ReadVariableOp?
StatsModel/dense_3/MatMul_8MatMul'StatsModel/dense_2/Relu_8:activations:02StatsModel/dense_3/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_8?
+StatsModel/dense_3/BiasAdd_8/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_8/ReadVariableOp?
StatsModel/dense_3/BiasAdd_8BiasAdd%StatsModel/dense_3/MatMul_8:product:03StatsModel/dense_3/BiasAdd_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_8?
*StatsModel/dense_2/MatMul_9/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_9/ReadVariableOp?
StatsModel/dense_2/MatMul_9MatMultf.unstack_1/unstack:output:42StatsModel/dense_2/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_9?
+StatsModel/dense_2/BiasAdd_9/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_9/ReadVariableOp?
StatsModel/dense_2/BiasAdd_9BiasAdd%StatsModel/dense_2/MatMul_9:product:03StatsModel/dense_2/BiasAdd_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_9?
StatsModel/dense_2/Relu_9Relu%StatsModel/dense_2/BiasAdd_9:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_9?
*StatsModel/dense_3/MatMul_9/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_9/ReadVariableOp?
StatsModel/dense_3/MatMul_9MatMul'StatsModel/dense_2/Relu_9:activations:02StatsModel/dense_3/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_9?
+StatsModel/dense_3/BiasAdd_9/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_9/ReadVariableOp?
StatsModel/dense_3/BiasAdd_9BiasAdd%StatsModel/dense_3/MatMul_9:product:03StatsModel/dense_3/BiasAdd_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMultf.concat_4/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0*^StatsModel/dense_2/BiasAdd/ReadVariableOp,^StatsModel/dense_2/BiasAdd_1/ReadVariableOp,^StatsModel/dense_2/BiasAdd_2/ReadVariableOp,^StatsModel/dense_2/BiasAdd_3/ReadVariableOp,^StatsModel/dense_2/BiasAdd_4/ReadVariableOp,^StatsModel/dense_2/BiasAdd_5/ReadVariableOp,^StatsModel/dense_2/BiasAdd_6/ReadVariableOp,^StatsModel/dense_2/BiasAdd_7/ReadVariableOp,^StatsModel/dense_2/BiasAdd_8/ReadVariableOp,^StatsModel/dense_2/BiasAdd_9/ReadVariableOp)^StatsModel/dense_2/MatMul/ReadVariableOp+^StatsModel/dense_2/MatMul_1/ReadVariableOp+^StatsModel/dense_2/MatMul_2/ReadVariableOp+^StatsModel/dense_2/MatMul_3/ReadVariableOp+^StatsModel/dense_2/MatMul_4/ReadVariableOp+^StatsModel/dense_2/MatMul_5/ReadVariableOp+^StatsModel/dense_2/MatMul_6/ReadVariableOp+^StatsModel/dense_2/MatMul_7/ReadVariableOp+^StatsModel/dense_2/MatMul_8/ReadVariableOp+^StatsModel/dense_2/MatMul_9/ReadVariableOp*^StatsModel/dense_3/BiasAdd/ReadVariableOp,^StatsModel/dense_3/BiasAdd_1/ReadVariableOp,^StatsModel/dense_3/BiasAdd_2/ReadVariableOp,^StatsModel/dense_3/BiasAdd_3/ReadVariableOp,^StatsModel/dense_3/BiasAdd_4/ReadVariableOp,^StatsModel/dense_3/BiasAdd_5/ReadVariableOp,^StatsModel/dense_3/BiasAdd_6/ReadVariableOp,^StatsModel/dense_3/BiasAdd_7/ReadVariableOp,^StatsModel/dense_3/BiasAdd_8/ReadVariableOp,^StatsModel/dense_3/BiasAdd_9/ReadVariableOp)^StatsModel/dense_3/MatMul/ReadVariableOp+^StatsModel/dense_3/MatMul_1/ReadVariableOp+^StatsModel/dense_3/MatMul_2/ReadVariableOp+^StatsModel/dense_3/MatMul_3/ReadVariableOp+^StatsModel/dense_3/MatMul_4/ReadVariableOp+^StatsModel/dense_3/MatMul_5/ReadVariableOp+^StatsModel/dense_3/MatMul_6/ReadVariableOp+^StatsModel/dense_3/MatMul_7/ReadVariableOp+^StatsModel/dense_3/MatMul_8/ReadVariableOp+^StatsModel/dense_3/MatMul_9/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2V
)StatsModel/dense_2/BiasAdd/ReadVariableOp)StatsModel/dense_2/BiasAdd/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_1/ReadVariableOp+StatsModel/dense_2/BiasAdd_1/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_2/ReadVariableOp+StatsModel/dense_2/BiasAdd_2/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_3/ReadVariableOp+StatsModel/dense_2/BiasAdd_3/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_4/ReadVariableOp+StatsModel/dense_2/BiasAdd_4/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_5/ReadVariableOp+StatsModel/dense_2/BiasAdd_5/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_6/ReadVariableOp+StatsModel/dense_2/BiasAdd_6/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_7/ReadVariableOp+StatsModel/dense_2/BiasAdd_7/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_8/ReadVariableOp+StatsModel/dense_2/BiasAdd_8/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_9/ReadVariableOp+StatsModel/dense_2/BiasAdd_9/ReadVariableOp2T
(StatsModel/dense_2/MatMul/ReadVariableOp(StatsModel/dense_2/MatMul/ReadVariableOp2X
*StatsModel/dense_2/MatMul_1/ReadVariableOp*StatsModel/dense_2/MatMul_1/ReadVariableOp2X
*StatsModel/dense_2/MatMul_2/ReadVariableOp*StatsModel/dense_2/MatMul_2/ReadVariableOp2X
*StatsModel/dense_2/MatMul_3/ReadVariableOp*StatsModel/dense_2/MatMul_3/ReadVariableOp2X
*StatsModel/dense_2/MatMul_4/ReadVariableOp*StatsModel/dense_2/MatMul_4/ReadVariableOp2X
*StatsModel/dense_2/MatMul_5/ReadVariableOp*StatsModel/dense_2/MatMul_5/ReadVariableOp2X
*StatsModel/dense_2/MatMul_6/ReadVariableOp*StatsModel/dense_2/MatMul_6/ReadVariableOp2X
*StatsModel/dense_2/MatMul_7/ReadVariableOp*StatsModel/dense_2/MatMul_7/ReadVariableOp2X
*StatsModel/dense_2/MatMul_8/ReadVariableOp*StatsModel/dense_2/MatMul_8/ReadVariableOp2X
*StatsModel/dense_2/MatMul_9/ReadVariableOp*StatsModel/dense_2/MatMul_9/ReadVariableOp2V
)StatsModel/dense_3/BiasAdd/ReadVariableOp)StatsModel/dense_3/BiasAdd/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_1/ReadVariableOp+StatsModel/dense_3/BiasAdd_1/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_2/ReadVariableOp+StatsModel/dense_3/BiasAdd_2/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_3/ReadVariableOp+StatsModel/dense_3/BiasAdd_3/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_4/ReadVariableOp+StatsModel/dense_3/BiasAdd_4/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_5/ReadVariableOp+StatsModel/dense_3/BiasAdd_5/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_6/ReadVariableOp+StatsModel/dense_3/BiasAdd_6/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_7/ReadVariableOp+StatsModel/dense_3/BiasAdd_7/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_8/ReadVariableOp+StatsModel/dense_3/BiasAdd_8/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_9/ReadVariableOp+StatsModel/dense_3/BiasAdd_9/ReadVariableOp2T
(StatsModel/dense_3/MatMul/ReadVariableOp(StatsModel/dense_3/MatMul/ReadVariableOp2X
*StatsModel/dense_3/MatMul_1/ReadVariableOp*StatsModel/dense_3/MatMul_1/ReadVariableOp2X
*StatsModel/dense_3/MatMul_2/ReadVariableOp*StatsModel/dense_3/MatMul_2/ReadVariableOp2X
*StatsModel/dense_3/MatMul_3/ReadVariableOp*StatsModel/dense_3/MatMul_3/ReadVariableOp2X
*StatsModel/dense_3/MatMul_4/ReadVariableOp*StatsModel/dense_3/MatMul_4/ReadVariableOp2X
*StatsModel/dense_3/MatMul_5/ReadVariableOp*StatsModel/dense_3/MatMul_5/ReadVariableOp2X
*StatsModel/dense_3/MatMul_6/ReadVariableOp*StatsModel/dense_3/MatMul_6/ReadVariableOp2X
*StatsModel/dense_3/MatMul_7/ReadVariableOp*StatsModel/dense_3/MatMul_7/ReadVariableOp2X
*StatsModel/dense_3/MatMul_8/ReadVariableOp*StatsModel/dense_3/MatMul_8/ReadVariableOp2X
*StatsModel/dense_3/MatMul_9/ReadVariableOp*StatsModel/dense_3/MatMul_9/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
D__inference_StatsModel_layer_call_and_return_conditional_losses_9061
dense_2_input 
dense_2_9050:
??
dense_2_9052:	?
dense_3_9055:	?
dense_3_9057:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_9050dense_2_9052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_89262!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_9055dense_3_9057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_89422!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_2_input
?
?
'__inference_dense_4_layer_call_fn_10084

inputs
unknown:	?d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_91752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_8926

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_10134

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?y
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9631
input_2#
statsmodel_9534:
??
statsmodel_9536:	?"
statsmodel_9538:	?
statsmodel_9540:
dense_4_9620:	?d
dense_4_9622:d
dense_5_9625:d
dense_5_9627:
identity??"StatsModel/StatefulPartitionedCall?$StatsModel/StatefulPartitionedCall_1?$StatsModel/StatefulPartitionedCall_2?$StatsModel/StatefulPartitionedCall_3?$StatsModel/StatefulPartitionedCall_4?$StatsModel/StatefulPartitionedCall_5?$StatsModel/StatefulPartitionedCall_6?$StatsModel/StatefulPartitionedCall_7?$StatsModel/StatefulPartitionedCall_8?$StatsModel/StatefulPartitionedCall_9?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
tf.unstack_1/unstackUnpackinput_2*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
"StatsModel/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_1/unstack:output:5statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092$
"StatsModel/StatefulPartitionedCall?
$StatsModel/StatefulPartitionedCall_1StatefulPartitionedCalltf.unstack_1/unstack:output:6statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_1?
$StatsModel/StatefulPartitionedCall_2StatefulPartitionedCalltf.unstack_1/unstack:output:7statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_2?
$StatsModel/StatefulPartitionedCall_3StatefulPartitionedCalltf.unstack_1/unstack:output:8statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_3?
$StatsModel/StatefulPartitionedCall_4StatefulPartitionedCalltf.unstack_1/unstack:output:9statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_4?
$StatsModel/StatefulPartitionedCall_5StatefulPartitionedCalltf.unstack_1/unstack:output:0statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_5?
$StatsModel/StatefulPartitionedCall_6StatefulPartitionedCalltf.unstack_1/unstack:output:1statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_6?
$StatsModel/StatefulPartitionedCall_7StatefulPartitionedCalltf.unstack_1/unstack:output:2statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_7?
$StatsModel/StatefulPartitionedCall_8StatefulPartitionedCalltf.unstack_1/unstack:output:3statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_8?
$StatsModel/StatefulPartitionedCall_9StatefulPartitionedCalltf.unstack_1/unstack:output:4statsmodel_9534statsmodel_9536statsmodel_9538statsmodel_9540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092&
$StatsModel/StatefulPartitionedCall_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_4_9620dense_4_9622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_91752!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9625dense_5_9627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_91922!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0#^StatsModel/StatefulPartitionedCall%^StatsModel/StatefulPartitionedCall_1%^StatsModel/StatefulPartitionedCall_2%^StatsModel/StatefulPartitionedCall_3%^StatsModel/StatefulPartitionedCall_4%^StatsModel/StatefulPartitionedCall_5%^StatsModel/StatefulPartitionedCall_6%^StatsModel/StatefulPartitionedCall_7%^StatsModel/StatefulPartitionedCall_8%^StatsModel/StatefulPartitionedCall_9 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2H
"StatsModel/StatefulPartitionedCall"StatsModel/StatefulPartitionedCall2L
$StatsModel/StatefulPartitionedCall_1$StatsModel/StatefulPartitionedCall_12L
$StatsModel/StatefulPartitionedCall_2$StatsModel/StatefulPartitionedCall_22L
$StatsModel/StatefulPartitionedCall_3$StatsModel/StatefulPartitionedCall_32L
$StatsModel/StatefulPartitionedCall_4$StatsModel/StatefulPartitionedCall_42L
$StatsModel/StatefulPartitionedCall_5$StatsModel/StatefulPartitionedCall_52L
$StatsModel/StatefulPartitionedCall_6$StatsModel/StatefulPartitionedCall_62L
$StatsModel/StatefulPartitionedCall_7$StatsModel/StatefulPartitionedCall_72L
$StatsModel/StatefulPartitionedCall_8$StatsModel/StatefulPartitionedCall_82L
$StatsModel/StatefulPartitionedCall_9$StatsModel/StatefulPartitionedCall_92B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?

?
A__inference_dense_5_layer_call_and_return_conditional_losses_9192

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_10392
file_prefix2
assignvariableop_dense_4_kernel:	?d-
assignvariableop_1_dense_4_bias:d3
!assignvariableop_2_dense_5_kernel:d-
assignvariableop_3_dense_5_bias:5
!assignvariableop_4_dense_2_kernel:
??.
assignvariableop_5_dense_2_bias:	?4
!assignvariableop_6_dense_3_kernel:	?-
assignvariableop_7_dense_3_bias:"
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 7
$assignvariableop_12_dense_4_kernel_m:	?d0
"assignvariableop_13_dense_4_bias_m:d6
$assignvariableop_14_dense_5_kernel_m:d0
"assignvariableop_15_dense_5_bias_m:8
$assignvariableop_16_dense_2_kernel_m:
??1
"assignvariableop_17_dense_2_bias_m:	?7
$assignvariableop_18_dense_3_kernel_m:	?0
"assignvariableop_19_dense_3_bias_m:7
$assignvariableop_20_dense_4_kernel_v:	?d0
"assignvariableop_21_dense_4_bias_v:d6
$assignvariableop_22_dense_5_kernel_v:d0
"assignvariableop_23_dense_5_bias_v:8
$assignvariableop_24_dense_2_kernel_v:
??1
"assignvariableop_25_dense_2_bias_v:	?7
$assignvariableop_26_dense_3_kernel_v:	?0
"assignvariableop_27_dense_3_bias_v::
&assignvariableop_28_dense_2_kernel_m_1:
??3
$assignvariableop_29_dense_2_bias_m_1:	?9
&assignvariableop_30_dense_3_kernel_m_1:	?2
$assignvariableop_31_dense_3_bias_m_1::
&assignvariableop_32_dense_2_kernel_v_1:
??3
$assignvariableop_33_dense_2_bias_v_1:	?9
&assignvariableop_34_dense_3_kernel_v_1:	?2
$assignvariableop_35_dense_3_bias_v_1:
identity_37??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/0/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_4_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_4_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dense_5_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_5_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_2_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_2_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_3_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_3_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_4_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_4_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_dense_5_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_5_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_dense_2_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_dense_2_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_dense_3_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_dense_3_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_dense_2_kernel_m_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp$assignvariableop_29_dense_2_bias_m_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_dense_3_kernel_m_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp$assignvariableop_31_dense_3_bias_m_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_dense_2_kernel_v_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_2_bias_v_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_dense_3_kernel_v_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp$assignvariableop_35_dense_3_bias_v_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_359
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36?
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_35AssignVariableOp_352(
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
?
?
)__inference_StatsModel_layer_call_fn_9033
dense_2_input
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_2_input
?	
?
(__inference_DualModel_layer_call_fn_9983

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:	?d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_DualModel_layer_call_and_return_conditional_losses_91992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
E__inference_StatsModel_layer_call_and_return_conditional_losses_10021

inputs:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_10115

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_3_layer_call_and_return_conditional_losses_8942

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_4_layer_call_and_return_conditional_losses_9175

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?y
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9199

inputs#
statsmodel_9078:
??
statsmodel_9080:	?"
statsmodel_9082:	?
statsmodel_9084:
dense_4_9176:	?d
dense_4_9178:d
dense_5_9193:d
dense_5_9195:
identity??"StatsModel/StatefulPartitionedCall?$StatsModel/StatefulPartitionedCall_1?$StatsModel/StatefulPartitionedCall_2?$StatsModel/StatefulPartitionedCall_3?$StatsModel/StatefulPartitionedCall_4?$StatsModel/StatefulPartitionedCall_5?$StatsModel/StatefulPartitionedCall_6?$StatsModel/StatefulPartitionedCall_7?$StatsModel/StatefulPartitionedCall_8?$StatsModel/StatefulPartitionedCall_9?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
tf.unstack_1/unstackUnpackinputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
"StatsModel/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_1/unstack:output:5statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492$
"StatsModel/StatefulPartitionedCall?
$StatsModel/StatefulPartitionedCall_1StatefulPartitionedCalltf.unstack_1/unstack:output:6statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_1?
$StatsModel/StatefulPartitionedCall_2StatefulPartitionedCalltf.unstack_1/unstack:output:7statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_2?
$StatsModel/StatefulPartitionedCall_3StatefulPartitionedCalltf.unstack_1/unstack:output:8statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_3?
$StatsModel/StatefulPartitionedCall_4StatefulPartitionedCalltf.unstack_1/unstack:output:9statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_4?
$StatsModel/StatefulPartitionedCall_5StatefulPartitionedCalltf.unstack_1/unstack:output:0statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_5?
$StatsModel/StatefulPartitionedCall_6StatefulPartitionedCalltf.unstack_1/unstack:output:1statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_6?
$StatsModel/StatefulPartitionedCall_7StatefulPartitionedCalltf.unstack_1/unstack:output:2statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_7?
$StatsModel/StatefulPartitionedCall_8StatefulPartitionedCalltf.unstack_1/unstack:output:3statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_8?
$StatsModel/StatefulPartitionedCall_9StatefulPartitionedCalltf.unstack_1/unstack:output:4statsmodel_9078statsmodel_9080statsmodel_9082statsmodel_9084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492&
$StatsModel/StatefulPartitionedCall_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack+StatsModel/StatefulPartitionedCall:output:0-StatsModel/StatefulPartitionedCall_1:output:0-StatsModel/StatefulPartitionedCall_2:output:0-StatsModel/StatefulPartitionedCall_3:output:0-StatsModel/StatefulPartitionedCall_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack-StatsModel/StatefulPartitionedCall_5:output:0-StatsModel/StatefulPartitionedCall_6:output:0-StatsModel/StatefulPartitionedCall_7:output:0-StatsModel/StatefulPartitionedCall_8:output:0-StatsModel/StatefulPartitionedCall_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_4_9176dense_4_9178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_91752!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9193dense_5_9195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_91922!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0#^StatsModel/StatefulPartitionedCall%^StatsModel/StatefulPartitionedCall_1%^StatsModel/StatefulPartitionedCall_2%^StatsModel/StatefulPartitionedCall_3%^StatsModel/StatefulPartitionedCall_4%^StatsModel/StatefulPartitionedCall_5%^StatsModel/StatefulPartitionedCall_6%^StatsModel/StatefulPartitionedCall_7%^StatsModel/StatefulPartitionedCall_8%^StatsModel/StatefulPartitionedCall_9 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2H
"StatsModel/StatefulPartitionedCall"StatsModel/StatefulPartitionedCall2L
$StatsModel/StatefulPartitionedCall_1$StatsModel/StatefulPartitionedCall_12L
$StatsModel/StatefulPartitionedCall_2$StatsModel/StatefulPartitionedCall_22L
$StatsModel/StatefulPartitionedCall_3$StatsModel/StatefulPartitionedCall_32L
$StatsModel/StatefulPartitionedCall_4$StatsModel/StatefulPartitionedCall_42L
$StatsModel/StatefulPartitionedCall_5$StatsModel/StatefulPartitionedCall_52L
$StatsModel/StatefulPartitionedCall_6$StatsModel/StatefulPartitionedCall_62L
$StatsModel/StatefulPartitionedCall_7$StatsModel/StatefulPartitionedCall_72L
$StatsModel/StatefulPartitionedCall_8$StatsModel/StatefulPartitionedCall_82L
$StatsModel/StatefulPartitionedCall_9$StatsModel/StatefulPartitionedCall_92B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?	
?
(__inference_DualModel_layer_call_fn_9218
input_2
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:	?d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_DualModel_layer_call_and_return_conditional_losses_91992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?
?
*__inference_StatsModel_layer_call_fn_10051

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_89492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_DualModel_layer_call_fn_10004

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:	?d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_DualModel_layer_call_and_return_conditional_losses_93712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_8908
input_2O
;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource:
??K
<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource:	?N
;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource:	?J
<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource:C
0dualmodel_dense_4_matmul_readvariableop_resource:	?d?
1dualmodel_dense_4_biasadd_readvariableop_resource:dB
0dualmodel_dense_5_matmul_readvariableop_resource:d?
1dualmodel_dense_5_biasadd_readvariableop_resource:
identity??3DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp?5DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp?2DualModel/StatsModel/dense_2/MatMul/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp?4DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp?3DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp?5DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp?2DualModel/StatsModel/dense_3/MatMul/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp?4DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp?(DualModel/dense_4/BiasAdd/ReadVariableOp?'DualModel/dense_4/MatMul/ReadVariableOp?(DualModel/dense_5/BiasAdd/ReadVariableOp?'DualModel/dense_5/MatMul/ReadVariableOp?
DualModel/tf.unstack_1/unstackUnpackinput_2*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2 
DualModel/tf.unstack_1/unstack?
2DualModel/StatsModel/dense_2/MatMul/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype024
2DualModel/StatsModel/dense_2/MatMul/ReadVariableOp?
#DualModel/StatsModel/dense_2/MatMulMatMul'DualModel/tf.unstack_1/unstack:output:5:DualModel/StatsModel/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/MatMul?
3DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp?
$DualModel/StatsModel/dense_2/BiasAddBiasAdd-DualModel/StatsModel/dense_2/MatMul:product:0;DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$DualModel/StatsModel/dense_2/BiasAdd?
!DualModel/StatsModel/dense_2/ReluRelu-DualModel/StatsModel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2#
!DualModel/StatsModel/dense_2/Relu?
2DualModel/StatsModel/dense_3/MatMul/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype024
2DualModel/StatsModel/dense_3/MatMul/ReadVariableOp?
#DualModel/StatsModel/dense_3/MatMulMatMul/DualModel/StatsModel/dense_2/Relu:activations:0:DualModel/StatsModel/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#DualModel/StatsModel/dense_3/MatMul?
3DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp?
$DualModel/StatsModel/dense_3/BiasAddBiasAdd-DualModel/StatsModel/dense_3/MatMul:product:0;DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$DualModel/StatsModel/dense_3/BiasAdd?
4DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_1MatMul'DualModel/tf.unstack_1/unstack:output:6<DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_1?
5DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_1BiasAdd/DualModel/StatsModel/dense_2/MatMul_1:product:0=DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_1?
#DualModel/StatsModel/dense_2/Relu_1Relu/DualModel/StatsModel/dense_2/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_1?
4DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_1MatMul1DualModel/StatsModel/dense_2/Relu_1:activations:0<DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_1?
5DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_1BiasAdd/DualModel/StatsModel/dense_3/MatMul_1:product:0=DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_1?
4DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_2MatMul'DualModel/tf.unstack_1/unstack:output:7<DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_2?
5DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_2BiasAdd/DualModel/StatsModel/dense_2/MatMul_2:product:0=DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_2?
#DualModel/StatsModel/dense_2/Relu_2Relu/DualModel/StatsModel/dense_2/BiasAdd_2:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_2?
4DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_2MatMul1DualModel/StatsModel/dense_2/Relu_2:activations:0<DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_2?
5DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_2BiasAdd/DualModel/StatsModel/dense_3/MatMul_2:product:0=DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_2?
4DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_3MatMul'DualModel/tf.unstack_1/unstack:output:8<DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_3?
5DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_3BiasAdd/DualModel/StatsModel/dense_2/MatMul_3:product:0=DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_3?
#DualModel/StatsModel/dense_2/Relu_3Relu/DualModel/StatsModel/dense_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_3?
4DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_3MatMul1DualModel/StatsModel/dense_2/Relu_3:activations:0<DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_3?
5DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_3BiasAdd/DualModel/StatsModel/dense_3/MatMul_3:product:0=DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_3?
4DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_4MatMul'DualModel/tf.unstack_1/unstack:output:9<DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_4?
5DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_4BiasAdd/DualModel/StatsModel/dense_2/MatMul_4:product:0=DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_4?
#DualModel/StatsModel/dense_2/Relu_4Relu/DualModel/StatsModel/dense_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_4?
4DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_4MatMul1DualModel/StatsModel/dense_2/Relu_4:activations:0<DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_4?
5DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_4BiasAdd/DualModel/StatsModel/dense_3/MatMul_4:product:0=DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_4?
4DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_5MatMul'DualModel/tf.unstack_1/unstack:output:0<DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_5?
5DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_5BiasAdd/DualModel/StatsModel/dense_2/MatMul_5:product:0=DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_5?
#DualModel/StatsModel/dense_2/Relu_5Relu/DualModel/StatsModel/dense_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_5?
4DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_5MatMul1DualModel/StatsModel/dense_2/Relu_5:activations:0<DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_5?
5DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_5BiasAdd/DualModel/StatsModel/dense_3/MatMul_5:product:0=DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_5?
4DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_6MatMul'DualModel/tf.unstack_1/unstack:output:1<DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_6?
5DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_6BiasAdd/DualModel/StatsModel/dense_2/MatMul_6:product:0=DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_6?
#DualModel/StatsModel/dense_2/Relu_6Relu/DualModel/StatsModel/dense_2/BiasAdd_6:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_6?
4DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_6MatMul1DualModel/StatsModel/dense_2/Relu_6:activations:0<DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_6?
5DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_6BiasAdd/DualModel/StatsModel/dense_3/MatMul_6:product:0=DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_6?
4DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_7MatMul'DualModel/tf.unstack_1/unstack:output:2<DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_7?
5DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_7BiasAdd/DualModel/StatsModel/dense_2/MatMul_7:product:0=DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_7?
#DualModel/StatsModel/dense_2/Relu_7Relu/DualModel/StatsModel/dense_2/BiasAdd_7:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_7?
4DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_7MatMul1DualModel/StatsModel/dense_2/Relu_7:activations:0<DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_7?
5DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_7BiasAdd/DualModel/StatsModel/dense_3/MatMul_7:product:0=DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_7?
4DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_8MatMul'DualModel/tf.unstack_1/unstack:output:3<DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_8?
5DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_8BiasAdd/DualModel/StatsModel/dense_2/MatMul_8:product:0=DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_8?
#DualModel/StatsModel/dense_2/Relu_8Relu/DualModel/StatsModel/dense_2/BiasAdd_8:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_8?
4DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_8MatMul1DualModel/StatsModel/dense_2/Relu_8:activations:0<DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_8?
5DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_8BiasAdd/DualModel/StatsModel/dense_3/MatMul_8:product:0=DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_8?
4DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp?
%DualModel/StatsModel/dense_2/MatMul_9MatMul'DualModel/tf.unstack_1/unstack:output:4<DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%DualModel/StatsModel/dense_2/MatMul_9?
5DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp?
&DualModel/StatsModel/dense_2/BiasAdd_9BiasAdd/DualModel/StatsModel/dense_2/MatMul_9:product:0=DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&DualModel/StatsModel/dense_2/BiasAdd_9?
#DualModel/StatsModel/dense_2/Relu_9Relu/DualModel/StatsModel/dense_2/BiasAdd_9:output:0*
T0*(
_output_shapes
:??????????2%
#DualModel/StatsModel/dense_2/Relu_9?
4DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOpReadVariableOp;dualmodel_statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp?
%DualModel/StatsModel/dense_3/MatMul_9MatMul1DualModel/StatsModel/dense_2/Relu_9:activations:0<DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%DualModel/StatsModel/dense_3/MatMul_9?
5DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOpReadVariableOp<dualmodel_statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp?
&DualModel/StatsModel/dense_3/BiasAdd_9BiasAdd/DualModel/StatsModel/dense_3/MatMul_9:product:0=DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&DualModel/StatsModel/dense_3/BiasAdd_9?
DualModel/tf.stack_3/stackPack'DualModel/tf.unstack_1/unstack:output:5'DualModel/tf.unstack_1/unstack:output:6'DualModel/tf.unstack_1/unstack:output:7'DualModel/tf.unstack_1/unstack:output:8'DualModel/tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
DualModel/tf.stack_3/stack?
DualModel/tf.stack_2/stackPack'DualModel/tf.unstack_1/unstack:output:0'DualModel/tf.unstack_1/unstack:output:1'DualModel/tf.unstack_1/unstack:output:2'DualModel/tf.unstack_1/unstack:output:3'DualModel/tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
DualModel/tf.stack_2/stack?
*DualModel/tf.math.reduce_mean_1/Mean/inputPack-DualModel/StatsModel/dense_3/BiasAdd:output:0/DualModel/StatsModel/dense_3/BiasAdd_1:output:0/DualModel/StatsModel/dense_3/BiasAdd_2:output:0/DualModel/StatsModel/dense_3/BiasAdd_3:output:0/DualModel/StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2,
*DualModel/tf.math.reduce_mean_1/Mean/input?
6DualModel/tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 28
6DualModel/tf.math.reduce_mean_1/Mean/reduction_indices?
$DualModel/tf.math.reduce_mean_1/MeanMean3DualModel/tf.math.reduce_mean_1/Mean/input:output:0?DualModel/tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2&
$DualModel/tf.math.reduce_mean_1/Mean?
(DualModel/tf.math.reduce_min_1/Min/inputPack-DualModel/StatsModel/dense_3/BiasAdd:output:0/DualModel/StatsModel/dense_3/BiasAdd_1:output:0/DualModel/StatsModel/dense_3/BiasAdd_2:output:0/DualModel/StatsModel/dense_3/BiasAdd_3:output:0/DualModel/StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2*
(DualModel/tf.math.reduce_min_1/Min/input?
4DualModel/tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4DualModel/tf.math.reduce_min_1/Min/reduction_indices?
"DualModel/tf.math.reduce_min_1/MinMin1DualModel/tf.math.reduce_min_1/Min/input:output:0=DualModel/tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2$
"DualModel/tf.math.reduce_min_1/Min?
(DualModel/tf.math.reduce_max_1/Max/inputPack-DualModel/StatsModel/dense_3/BiasAdd:output:0/DualModel/StatsModel/dense_3/BiasAdd_1:output:0/DualModel/StatsModel/dense_3/BiasAdd_2:output:0/DualModel/StatsModel/dense_3/BiasAdd_3:output:0/DualModel/StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2*
(DualModel/tf.math.reduce_max_1/Max/input?
4DualModel/tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4DualModel/tf.math.reduce_max_1/Max/reduction_indices?
"DualModel/tf.math.reduce_max_1/MaxMax1DualModel/tf.math.reduce_max_1/Max/input:output:0=DualModel/tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2$
"DualModel/tf.math.reduce_max_1/Max?
(DualModel/tf.math.reduce_mean/Mean/inputPack/DualModel/StatsModel/dense_3/BiasAdd_5:output:0/DualModel/StatsModel/dense_3/BiasAdd_6:output:0/DualModel/StatsModel/dense_3/BiasAdd_7:output:0/DualModel/StatsModel/dense_3/BiasAdd_8:output:0/DualModel/StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2*
(DualModel/tf.math.reduce_mean/Mean/input?
4DualModel/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4DualModel/tf.math.reduce_mean/Mean/reduction_indices?
"DualModel/tf.math.reduce_mean/MeanMean1DualModel/tf.math.reduce_mean/Mean/input:output:0=DualModel/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2$
"DualModel/tf.math.reduce_mean/Mean?
&DualModel/tf.math.reduce_min/Min/inputPack/DualModel/StatsModel/dense_3/BiasAdd_5:output:0/DualModel/StatsModel/dense_3/BiasAdd_6:output:0/DualModel/StatsModel/dense_3/BiasAdd_7:output:0/DualModel/StatsModel/dense_3/BiasAdd_8:output:0/DualModel/StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2(
&DualModel/tf.math.reduce_min/Min/input?
2DualModel/tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2DualModel/tf.math.reduce_min/Min/reduction_indices?
 DualModel/tf.math.reduce_min/MinMin/DualModel/tf.math.reduce_min/Min/input:output:0;DualModel/tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 DualModel/tf.math.reduce_min/Min?
&DualModel/tf.math.reduce_max/Max/inputPack/DualModel/StatsModel/dense_3/BiasAdd_5:output:0/DualModel/StatsModel/dense_3/BiasAdd_6:output:0/DualModel/StatsModel/dense_3/BiasAdd_7:output:0/DualModel/StatsModel/dense_3/BiasAdd_8:output:0/DualModel/StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2(
&DualModel/tf.math.reduce_max/Max/input?
2DualModel/tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2DualModel/tf.math.reduce_max/Max/reduction_indices?
 DualModel/tf.math.reduce_max/MaxMax/DualModel/tf.math.reduce_max/Max/input:output:0;DualModel/tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 DualModel/tf.math.reduce_max/Max?
4DualModel/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4DualModel/tf.math.reduce_sum_2/Sum/reduction_indices?
"DualModel/tf.math.reduce_sum_2/SumSum#DualModel/tf.stack_2/stack:output:0=DualModel/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"DualModel/tf.math.reduce_sum_2/Sum?
4DualModel/tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4DualModel/tf.math.reduce_sum_3/Sum/reduction_indices?
"DualModel/tf.math.reduce_sum_3/SumSum#DualModel/tf.stack_3/stack:output:0=DualModel/tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"DualModel/tf.math.reduce_sum_3/Sum?
!DualModel/tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!DualModel/tf.concat_3/concat/axis?
DualModel/tf.concat_3/concatConcatV2+DualModel/tf.math.reduce_sum_2/Sum:output:0+DualModel/tf.math.reduce_sum_3/Sum:output:0*DualModel/tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
DualModel/tf.concat_3/concat?
!DualModel/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!DualModel/tf.concat_1/concat/axis?
DualModel/tf.concat_1/concatConcatV2+DualModel/tf.math.reduce_mean/Mean:output:0)DualModel/tf.math.reduce_min/Min:output:0)DualModel/tf.math.reduce_max/Max:output:0*DualModel/tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
DualModel/tf.concat_1/concat?
!DualModel/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!DualModel/tf.concat_2/concat/axis?
DualModel/tf.concat_2/concatConcatV2-DualModel/tf.math.reduce_mean_1/Mean:output:0+DualModel/tf.math.reduce_min_1/Min:output:0+DualModel/tf.math.reduce_max_1/Max:output:0*DualModel/tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
DualModel/tf.concat_2/concat?
!DualModel/tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!DualModel/tf.concat_4/concat/axis?
DualModel/tf.concat_4/concatConcatV2%DualModel/tf.concat_3/concat:output:0%DualModel/tf.concat_1/concat:output:0%DualModel/tf.concat_2/concat:output:0*DualModel/tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
DualModel/tf.concat_4/concat?
'DualModel/dense_4/MatMul/ReadVariableOpReadVariableOp0dualmodel_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02)
'DualModel/dense_4/MatMul/ReadVariableOp?
DualModel/dense_4/MatMulMatMul%DualModel/tf.concat_4/concat:output:0/DualModel/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
DualModel/dense_4/MatMul?
(DualModel/dense_4/BiasAdd/ReadVariableOpReadVariableOp1dualmodel_dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02*
(DualModel/dense_4/BiasAdd/ReadVariableOp?
DualModel/dense_4/BiasAddBiasAdd"DualModel/dense_4/MatMul:product:00DualModel/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
DualModel/dense_4/BiasAdd?
DualModel/dense_4/ReluRelu"DualModel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
DualModel/dense_4/Relu?
'DualModel/dense_5/MatMul/ReadVariableOpReadVariableOp0dualmodel_dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02)
'DualModel/dense_5/MatMul/ReadVariableOp?
DualModel/dense_5/MatMulMatMul$DualModel/dense_4/Relu:activations:0/DualModel/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DualModel/dense_5/MatMul?
(DualModel/dense_5/BiasAdd/ReadVariableOpReadVariableOp1dualmodel_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(DualModel/dense_5/BiasAdd/ReadVariableOp?
DualModel/dense_5/BiasAddBiasAdd"DualModel/dense_5/MatMul:product:00DualModel/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DualModel/dense_5/BiasAdd?
DualModel/dense_5/SigmoidSigmoid"DualModel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
DualModel/dense_5/Sigmoid?
IdentityIdentityDualModel/dense_5/Sigmoid:y:04^DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp6^DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp3^DualModel/StatsModel/dense_2/MatMul/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp5^DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp4^DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp6^DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp3^DualModel/StatsModel/dense_3/MatMul/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp5^DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp)^DualModel/dense_4/BiasAdd/ReadVariableOp(^DualModel/dense_4/MatMul/ReadVariableOp)^DualModel/dense_5/BiasAdd/ReadVariableOp(^DualModel/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2j
3DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp3DualModel/StatsModel/dense_2/BiasAdd/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_1/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_2/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_3/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_4/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_5/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_6/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_7/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_8/ReadVariableOp2n
5DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp5DualModel/StatsModel/dense_2/BiasAdd_9/ReadVariableOp2h
2DualModel/StatsModel/dense_2/MatMul/ReadVariableOp2DualModel/StatsModel/dense_2/MatMul/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_1/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_2/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_3/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_4/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_5/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_6/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_7/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_8/ReadVariableOp2l
4DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp4DualModel/StatsModel/dense_2/MatMul_9/ReadVariableOp2j
3DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp3DualModel/StatsModel/dense_3/BiasAdd/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_1/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_2/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_3/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_4/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_5/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_6/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_7/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_8/ReadVariableOp2n
5DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp5DualModel/StatsModel/dense_3/BiasAdd_9/ReadVariableOp2h
2DualModel/StatsModel/dense_3/MatMul/ReadVariableOp2DualModel/StatsModel/dense_3/MatMul/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_1/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_2/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_3/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_4/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_5/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_6/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_7/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_8/ReadVariableOp2l
4DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp4DualModel/StatsModel/dense_3/MatMul_9/ReadVariableOp2T
(DualModel/dense_4/BiasAdd/ReadVariableOp(DualModel/dense_4/BiasAdd/ReadVariableOp2R
'DualModel/dense_4/MatMul/ReadVariableOp'DualModel/dense_4/MatMul/ReadVariableOp2T
(DualModel/dense_5/BiasAdd/ReadVariableOp(DualModel/dense_5/BiasAdd/ReadVariableOp2R
'DualModel/dense_5/MatMul/ReadVariableOp'DualModel/dense_5/MatMul/ReadVariableOp:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_10075

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_9654
input_2
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:	?d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_89082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?
?
E__inference_StatsModel_layer_call_and_return_conditional_losses_10038

inputs:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
(__inference_DualModel_layer_call_fn_9411
input_2
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
	unknown_3:	?d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_DualModel_layer_call_and_return_conditional_losses_93712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????
?
!
_user_specified_name	input_2
?
?
D__inference_StatsModel_layer_call_and_return_conditional_losses_8949

inputs 
dense_2_8927:
??
dense_2_8929:	?
dense_3_8943:	?
dense_3_8945:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_8927dense_2_8929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_89262!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_8943dense_3_8945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_89422!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_StatsModel_layer_call_and_return_conditional_losses_9009

inputs 
dense_2_8998:
??
dense_2_9000:	?
dense_3_9003:	?
dense_3_9005:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_8998dense_2_9000*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_89262!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_9003dense_3_9005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_89422!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
̖
?
C__inference_DualModel_layer_call_and_return_conditional_losses_9962

inputsE
1statsmodel_dense_2_matmul_readvariableop_resource:
??A
2statsmodel_dense_2_biasadd_readvariableop_resource:	?D
1statsmodel_dense_3_matmul_readvariableop_resource:	?@
2statsmodel_dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?d5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:d5
'dense_5_biasadd_readvariableop_resource:
identity??)StatsModel/dense_2/BiasAdd/ReadVariableOp?+StatsModel/dense_2/BiasAdd_1/ReadVariableOp?+StatsModel/dense_2/BiasAdd_2/ReadVariableOp?+StatsModel/dense_2/BiasAdd_3/ReadVariableOp?+StatsModel/dense_2/BiasAdd_4/ReadVariableOp?+StatsModel/dense_2/BiasAdd_5/ReadVariableOp?+StatsModel/dense_2/BiasAdd_6/ReadVariableOp?+StatsModel/dense_2/BiasAdd_7/ReadVariableOp?+StatsModel/dense_2/BiasAdd_8/ReadVariableOp?+StatsModel/dense_2/BiasAdd_9/ReadVariableOp?(StatsModel/dense_2/MatMul/ReadVariableOp?*StatsModel/dense_2/MatMul_1/ReadVariableOp?*StatsModel/dense_2/MatMul_2/ReadVariableOp?*StatsModel/dense_2/MatMul_3/ReadVariableOp?*StatsModel/dense_2/MatMul_4/ReadVariableOp?*StatsModel/dense_2/MatMul_5/ReadVariableOp?*StatsModel/dense_2/MatMul_6/ReadVariableOp?*StatsModel/dense_2/MatMul_7/ReadVariableOp?*StatsModel/dense_2/MatMul_8/ReadVariableOp?*StatsModel/dense_2/MatMul_9/ReadVariableOp?)StatsModel/dense_3/BiasAdd/ReadVariableOp?+StatsModel/dense_3/BiasAdd_1/ReadVariableOp?+StatsModel/dense_3/BiasAdd_2/ReadVariableOp?+StatsModel/dense_3/BiasAdd_3/ReadVariableOp?+StatsModel/dense_3/BiasAdd_4/ReadVariableOp?+StatsModel/dense_3/BiasAdd_5/ReadVariableOp?+StatsModel/dense_3/BiasAdd_6/ReadVariableOp?+StatsModel/dense_3/BiasAdd_7/ReadVariableOp?+StatsModel/dense_3/BiasAdd_8/ReadVariableOp?+StatsModel/dense_3/BiasAdd_9/ReadVariableOp?(StatsModel/dense_3/MatMul/ReadVariableOp?*StatsModel/dense_3/MatMul_1/ReadVariableOp?*StatsModel/dense_3/MatMul_2/ReadVariableOp?*StatsModel/dense_3/MatMul_3/ReadVariableOp?*StatsModel/dense_3/MatMul_4/ReadVariableOp?*StatsModel/dense_3/MatMul_5/ReadVariableOp?*StatsModel/dense_3/MatMul_6/ReadVariableOp?*StatsModel/dense_3/MatMul_7/ReadVariableOp?*StatsModel/dense_3/MatMul_8/ReadVariableOp?*StatsModel/dense_3/MatMul_9/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
tf.unstack_1/unstackUnpackinputs*
T0*?
_output_shapes?
?:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????:??????????*

axis*	
num
2
tf.unstack_1/unstack?
(StatsModel/dense_2/MatMul/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(StatsModel/dense_2/MatMul/ReadVariableOp?
StatsModel/dense_2/MatMulMatMultf.unstack_1/unstack:output:50StatsModel/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul?
)StatsModel/dense_2/BiasAdd/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)StatsModel/dense_2/BiasAdd/ReadVariableOp?
StatsModel/dense_2/BiasAddBiasAdd#StatsModel/dense_2/MatMul:product:01StatsModel/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd?
StatsModel/dense_2/ReluRelu#StatsModel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu?
(StatsModel/dense_3/MatMul/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(StatsModel/dense_3/MatMul/ReadVariableOp?
StatsModel/dense_3/MatMulMatMul%StatsModel/dense_2/Relu:activations:00StatsModel/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul?
)StatsModel/dense_3/BiasAdd/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)StatsModel/dense_3/BiasAdd/ReadVariableOp?
StatsModel/dense_3/BiasAddBiasAdd#StatsModel/dense_3/MatMul:product:01StatsModel/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd?
*StatsModel/dense_2/MatMul_1/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_1/ReadVariableOp?
StatsModel/dense_2/MatMul_1MatMultf.unstack_1/unstack:output:62StatsModel/dense_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_1?
+StatsModel/dense_2/BiasAdd_1/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_1/ReadVariableOp?
StatsModel/dense_2/BiasAdd_1BiasAdd%StatsModel/dense_2/MatMul_1:product:03StatsModel/dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_1?
StatsModel/dense_2/Relu_1Relu%StatsModel/dense_2/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_1?
*StatsModel/dense_3/MatMul_1/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_1/ReadVariableOp?
StatsModel/dense_3/MatMul_1MatMul'StatsModel/dense_2/Relu_1:activations:02StatsModel/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_1?
+StatsModel/dense_3/BiasAdd_1/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_1/ReadVariableOp?
StatsModel/dense_3/BiasAdd_1BiasAdd%StatsModel/dense_3/MatMul_1:product:03StatsModel/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_1?
*StatsModel/dense_2/MatMul_2/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_2/ReadVariableOp?
StatsModel/dense_2/MatMul_2MatMultf.unstack_1/unstack:output:72StatsModel/dense_2/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_2?
+StatsModel/dense_2/BiasAdd_2/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_2/ReadVariableOp?
StatsModel/dense_2/BiasAdd_2BiasAdd%StatsModel/dense_2/MatMul_2:product:03StatsModel/dense_2/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_2?
StatsModel/dense_2/Relu_2Relu%StatsModel/dense_2/BiasAdd_2:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_2?
*StatsModel/dense_3/MatMul_2/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_2/ReadVariableOp?
StatsModel/dense_3/MatMul_2MatMul'StatsModel/dense_2/Relu_2:activations:02StatsModel/dense_3/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_2?
+StatsModel/dense_3/BiasAdd_2/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_2/ReadVariableOp?
StatsModel/dense_3/BiasAdd_2BiasAdd%StatsModel/dense_3/MatMul_2:product:03StatsModel/dense_3/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_2?
*StatsModel/dense_2/MatMul_3/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_3/ReadVariableOp?
StatsModel/dense_2/MatMul_3MatMultf.unstack_1/unstack:output:82StatsModel/dense_2/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_3?
+StatsModel/dense_2/BiasAdd_3/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_3/ReadVariableOp?
StatsModel/dense_2/BiasAdd_3BiasAdd%StatsModel/dense_2/MatMul_3:product:03StatsModel/dense_2/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_3?
StatsModel/dense_2/Relu_3Relu%StatsModel/dense_2/BiasAdd_3:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_3?
*StatsModel/dense_3/MatMul_3/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_3/ReadVariableOp?
StatsModel/dense_3/MatMul_3MatMul'StatsModel/dense_2/Relu_3:activations:02StatsModel/dense_3/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_3?
+StatsModel/dense_3/BiasAdd_3/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_3/ReadVariableOp?
StatsModel/dense_3/BiasAdd_3BiasAdd%StatsModel/dense_3/MatMul_3:product:03StatsModel/dense_3/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_3?
*StatsModel/dense_2/MatMul_4/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_4/ReadVariableOp?
StatsModel/dense_2/MatMul_4MatMultf.unstack_1/unstack:output:92StatsModel/dense_2/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_4?
+StatsModel/dense_2/BiasAdd_4/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_4/ReadVariableOp?
StatsModel/dense_2/BiasAdd_4BiasAdd%StatsModel/dense_2/MatMul_4:product:03StatsModel/dense_2/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_4?
StatsModel/dense_2/Relu_4Relu%StatsModel/dense_2/BiasAdd_4:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_4?
*StatsModel/dense_3/MatMul_4/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_4/ReadVariableOp?
StatsModel/dense_3/MatMul_4MatMul'StatsModel/dense_2/Relu_4:activations:02StatsModel/dense_3/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_4?
+StatsModel/dense_3/BiasAdd_4/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_4/ReadVariableOp?
StatsModel/dense_3/BiasAdd_4BiasAdd%StatsModel/dense_3/MatMul_4:product:03StatsModel/dense_3/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_4?
*StatsModel/dense_2/MatMul_5/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_5/ReadVariableOp?
StatsModel/dense_2/MatMul_5MatMultf.unstack_1/unstack:output:02StatsModel/dense_2/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_5?
+StatsModel/dense_2/BiasAdd_5/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_5/ReadVariableOp?
StatsModel/dense_2/BiasAdd_5BiasAdd%StatsModel/dense_2/MatMul_5:product:03StatsModel/dense_2/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_5?
StatsModel/dense_2/Relu_5Relu%StatsModel/dense_2/BiasAdd_5:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_5?
*StatsModel/dense_3/MatMul_5/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_5/ReadVariableOp?
StatsModel/dense_3/MatMul_5MatMul'StatsModel/dense_2/Relu_5:activations:02StatsModel/dense_3/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_5?
+StatsModel/dense_3/BiasAdd_5/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_5/ReadVariableOp?
StatsModel/dense_3/BiasAdd_5BiasAdd%StatsModel/dense_3/MatMul_5:product:03StatsModel/dense_3/BiasAdd_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_5?
*StatsModel/dense_2/MatMul_6/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_6/ReadVariableOp?
StatsModel/dense_2/MatMul_6MatMultf.unstack_1/unstack:output:12StatsModel/dense_2/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_6?
+StatsModel/dense_2/BiasAdd_6/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_6/ReadVariableOp?
StatsModel/dense_2/BiasAdd_6BiasAdd%StatsModel/dense_2/MatMul_6:product:03StatsModel/dense_2/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_6?
StatsModel/dense_2/Relu_6Relu%StatsModel/dense_2/BiasAdd_6:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_6?
*StatsModel/dense_3/MatMul_6/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_6/ReadVariableOp?
StatsModel/dense_3/MatMul_6MatMul'StatsModel/dense_2/Relu_6:activations:02StatsModel/dense_3/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_6?
+StatsModel/dense_3/BiasAdd_6/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_6/ReadVariableOp?
StatsModel/dense_3/BiasAdd_6BiasAdd%StatsModel/dense_3/MatMul_6:product:03StatsModel/dense_3/BiasAdd_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_6?
*StatsModel/dense_2/MatMul_7/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_7/ReadVariableOp?
StatsModel/dense_2/MatMul_7MatMultf.unstack_1/unstack:output:22StatsModel/dense_2/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_7?
+StatsModel/dense_2/BiasAdd_7/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_7/ReadVariableOp?
StatsModel/dense_2/BiasAdd_7BiasAdd%StatsModel/dense_2/MatMul_7:product:03StatsModel/dense_2/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_7?
StatsModel/dense_2/Relu_7Relu%StatsModel/dense_2/BiasAdd_7:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_7?
*StatsModel/dense_3/MatMul_7/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_7/ReadVariableOp?
StatsModel/dense_3/MatMul_7MatMul'StatsModel/dense_2/Relu_7:activations:02StatsModel/dense_3/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_7?
+StatsModel/dense_3/BiasAdd_7/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_7/ReadVariableOp?
StatsModel/dense_3/BiasAdd_7BiasAdd%StatsModel/dense_3/MatMul_7:product:03StatsModel/dense_3/BiasAdd_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_7?
*StatsModel/dense_2/MatMul_8/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_8/ReadVariableOp?
StatsModel/dense_2/MatMul_8MatMultf.unstack_1/unstack:output:32StatsModel/dense_2/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_8?
+StatsModel/dense_2/BiasAdd_8/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_8/ReadVariableOp?
StatsModel/dense_2/BiasAdd_8BiasAdd%StatsModel/dense_2/MatMul_8:product:03StatsModel/dense_2/BiasAdd_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_8?
StatsModel/dense_2/Relu_8Relu%StatsModel/dense_2/BiasAdd_8:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_8?
*StatsModel/dense_3/MatMul_8/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_8/ReadVariableOp?
StatsModel/dense_3/MatMul_8MatMul'StatsModel/dense_2/Relu_8:activations:02StatsModel/dense_3/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_8?
+StatsModel/dense_3/BiasAdd_8/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_8/ReadVariableOp?
StatsModel/dense_3/BiasAdd_8BiasAdd%StatsModel/dense_3/MatMul_8:product:03StatsModel/dense_3/BiasAdd_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_8?
*StatsModel/dense_2/MatMul_9/ReadVariableOpReadVariableOp1statsmodel_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*StatsModel/dense_2/MatMul_9/ReadVariableOp?
StatsModel/dense_2/MatMul_9MatMultf.unstack_1/unstack:output:42StatsModel/dense_2/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/MatMul_9?
+StatsModel/dense_2/BiasAdd_9/ReadVariableOpReadVariableOp2statsmodel_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+StatsModel/dense_2/BiasAdd_9/ReadVariableOp?
StatsModel/dense_2/BiasAdd_9BiasAdd%StatsModel/dense_2/MatMul_9:product:03StatsModel/dense_2/BiasAdd_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/BiasAdd_9?
StatsModel/dense_2/Relu_9Relu%StatsModel/dense_2/BiasAdd_9:output:0*
T0*(
_output_shapes
:??????????2
StatsModel/dense_2/Relu_9?
*StatsModel/dense_3/MatMul_9/ReadVariableOpReadVariableOp1statsmodel_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*StatsModel/dense_3/MatMul_9/ReadVariableOp?
StatsModel/dense_3/MatMul_9MatMul'StatsModel/dense_2/Relu_9:activations:02StatsModel/dense_3/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/MatMul_9?
+StatsModel/dense_3/BiasAdd_9/ReadVariableOpReadVariableOp2statsmodel_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+StatsModel/dense_3/BiasAdd_9/ReadVariableOp?
StatsModel/dense_3/BiasAdd_9BiasAdd%StatsModel/dense_3/MatMul_9:product:03StatsModel/dense_3/BiasAdd_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
StatsModel/dense_3/BiasAdd_9?
tf.stack_3/stackPacktf.unstack_1/unstack:output:5tf.unstack_1/unstack:output:6tf.unstack_1/unstack:output:7tf.unstack_1/unstack:output:8tf.unstack_1/unstack:output:9*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_3/stack?
tf.stack_2/stackPacktf.unstack_1/unstack:output:0tf.unstack_1/unstack:output:1tf.unstack_1/unstack:output:2tf.unstack_1/unstack:output:3tf.unstack_1/unstack:output:4*
N*
T0*,
_output_shapes
:??????????*

axis2
tf.stack_2/stack?
 tf.math.reduce_mean_1/Mean/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2"
 tf.math.reduce_mean_1/Mean/input?
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,tf.math.reduce_mean_1/Mean/reduction_indices?
tf.math.reduce_mean_1/MeanMean)tf.math.reduce_mean_1/Mean/input:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean_1/Mean?
tf.math.reduce_min_1/Min/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_min_1/Min/input?
*tf.math.reduce_min_1/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_min_1/Min/reduction_indices?
tf.math.reduce_min_1/MinMin'tf.math.reduce_min_1/Min/input:output:03tf.math.reduce_min_1/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min_1/Min?
tf.math.reduce_max_1/Max/inputPack#StatsModel/dense_3/BiasAdd:output:0%StatsModel/dense_3/BiasAdd_1:output:0%StatsModel/dense_3/BiasAdd_2:output:0%StatsModel/dense_3/BiasAdd_3:output:0%StatsModel/dense_3/BiasAdd_4:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_max_1/Max/input?
*tf.math.reduce_max_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_max_1/Max/reduction_indices?
tf.math.reduce_max_1/MaxMax'tf.math.reduce_max_1/Max/input:output:03tf.math.reduce_max_1/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max_1/Max?
tf.math.reduce_mean/Mean/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2 
tf.math.reduce_mean/Mean/input?
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2,
*tf.math.reduce_mean/Mean/reduction_indices?
tf.math.reduce_mean/MeanMean'tf.math.reduce_mean/Mean/input:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_mean/Mean?
tf.math.reduce_min/Min/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_min/Min/input?
(tf.math.reduce_min/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_min/Min/reduction_indices?
tf.math.reduce_min/MinMin%tf.math.reduce_min/Min/input:output:01tf.math.reduce_min/Min/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_min/Min?
tf.math.reduce_max/Max/inputPack%StatsModel/dense_3/BiasAdd_5:output:0%StatsModel/dense_3/BiasAdd_6:output:0%StatsModel/dense_3/BiasAdd_7:output:0%StatsModel/dense_3/BiasAdd_8:output:0%StatsModel/dense_3/BiasAdd_9:output:0*
N*
T0*+
_output_shapes
:?????????2
tf.math.reduce_max/Max/input?
(tf.math.reduce_max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(tf.math.reduce_max/Max/reduction_indices?
tf.math.reduce_max/MaxMax%tf.math.reduce_max/Max/input:output:01tf.math.reduce_max/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
tf.math.reduce_max/Max?
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices?
tf.math.reduce_sum_2/SumSumtf.stack_2/stack:output:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_2/Sum?
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices?
tf.math.reduce_sum_3/SumSumtf.stack_3/stack:output:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_3/Sumt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axis?
tf.concat_3/concatConcatV2!tf.math.reduce_sum_2/Sum:output:0!tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_3/concatt
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2!tf.math.reduce_mean/Mean:output:0tf.math.reduce_min/Min:output:0tf.math.reduce_max/Max:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_1/concatt
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0!tf.math.reduce_min_1/Min:output:0!tf.math.reduce_max_1/Max:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????B2
tf.concat_2/concatt
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_4/concat/axis?
tf.concat_4/concatConcatV2tf.concat_3/concat:output:0tf.concat_1/concat:output:0tf.concat_2/concat:output:0 tf.concat_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_4/concat?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMultf.concat_4/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_4/Relu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Sigmoid?
IdentityIdentitydense_5/Sigmoid:y:0*^StatsModel/dense_2/BiasAdd/ReadVariableOp,^StatsModel/dense_2/BiasAdd_1/ReadVariableOp,^StatsModel/dense_2/BiasAdd_2/ReadVariableOp,^StatsModel/dense_2/BiasAdd_3/ReadVariableOp,^StatsModel/dense_2/BiasAdd_4/ReadVariableOp,^StatsModel/dense_2/BiasAdd_5/ReadVariableOp,^StatsModel/dense_2/BiasAdd_6/ReadVariableOp,^StatsModel/dense_2/BiasAdd_7/ReadVariableOp,^StatsModel/dense_2/BiasAdd_8/ReadVariableOp,^StatsModel/dense_2/BiasAdd_9/ReadVariableOp)^StatsModel/dense_2/MatMul/ReadVariableOp+^StatsModel/dense_2/MatMul_1/ReadVariableOp+^StatsModel/dense_2/MatMul_2/ReadVariableOp+^StatsModel/dense_2/MatMul_3/ReadVariableOp+^StatsModel/dense_2/MatMul_4/ReadVariableOp+^StatsModel/dense_2/MatMul_5/ReadVariableOp+^StatsModel/dense_2/MatMul_6/ReadVariableOp+^StatsModel/dense_2/MatMul_7/ReadVariableOp+^StatsModel/dense_2/MatMul_8/ReadVariableOp+^StatsModel/dense_2/MatMul_9/ReadVariableOp*^StatsModel/dense_3/BiasAdd/ReadVariableOp,^StatsModel/dense_3/BiasAdd_1/ReadVariableOp,^StatsModel/dense_3/BiasAdd_2/ReadVariableOp,^StatsModel/dense_3/BiasAdd_3/ReadVariableOp,^StatsModel/dense_3/BiasAdd_4/ReadVariableOp,^StatsModel/dense_3/BiasAdd_5/ReadVariableOp,^StatsModel/dense_3/BiasAdd_6/ReadVariableOp,^StatsModel/dense_3/BiasAdd_7/ReadVariableOp,^StatsModel/dense_3/BiasAdd_8/ReadVariableOp,^StatsModel/dense_3/BiasAdd_9/ReadVariableOp)^StatsModel/dense_3/MatMul/ReadVariableOp+^StatsModel/dense_3/MatMul_1/ReadVariableOp+^StatsModel/dense_3/MatMul_2/ReadVariableOp+^StatsModel/dense_3/MatMul_3/ReadVariableOp+^StatsModel/dense_3/MatMul_4/ReadVariableOp+^StatsModel/dense_3/MatMul_5/ReadVariableOp+^StatsModel/dense_3/MatMul_6/ReadVariableOp+^StatsModel/dense_3/MatMul_7/ReadVariableOp+^StatsModel/dense_3/MatMul_8/ReadVariableOp+^StatsModel/dense_3/MatMul_9/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????
?: : : : : : : : 2V
)StatsModel/dense_2/BiasAdd/ReadVariableOp)StatsModel/dense_2/BiasAdd/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_1/ReadVariableOp+StatsModel/dense_2/BiasAdd_1/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_2/ReadVariableOp+StatsModel/dense_2/BiasAdd_2/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_3/ReadVariableOp+StatsModel/dense_2/BiasAdd_3/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_4/ReadVariableOp+StatsModel/dense_2/BiasAdd_4/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_5/ReadVariableOp+StatsModel/dense_2/BiasAdd_5/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_6/ReadVariableOp+StatsModel/dense_2/BiasAdd_6/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_7/ReadVariableOp+StatsModel/dense_2/BiasAdd_7/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_8/ReadVariableOp+StatsModel/dense_2/BiasAdd_8/ReadVariableOp2Z
+StatsModel/dense_2/BiasAdd_9/ReadVariableOp+StatsModel/dense_2/BiasAdd_9/ReadVariableOp2T
(StatsModel/dense_2/MatMul/ReadVariableOp(StatsModel/dense_2/MatMul/ReadVariableOp2X
*StatsModel/dense_2/MatMul_1/ReadVariableOp*StatsModel/dense_2/MatMul_1/ReadVariableOp2X
*StatsModel/dense_2/MatMul_2/ReadVariableOp*StatsModel/dense_2/MatMul_2/ReadVariableOp2X
*StatsModel/dense_2/MatMul_3/ReadVariableOp*StatsModel/dense_2/MatMul_3/ReadVariableOp2X
*StatsModel/dense_2/MatMul_4/ReadVariableOp*StatsModel/dense_2/MatMul_4/ReadVariableOp2X
*StatsModel/dense_2/MatMul_5/ReadVariableOp*StatsModel/dense_2/MatMul_5/ReadVariableOp2X
*StatsModel/dense_2/MatMul_6/ReadVariableOp*StatsModel/dense_2/MatMul_6/ReadVariableOp2X
*StatsModel/dense_2/MatMul_7/ReadVariableOp*StatsModel/dense_2/MatMul_7/ReadVariableOp2X
*StatsModel/dense_2/MatMul_8/ReadVariableOp*StatsModel/dense_2/MatMul_8/ReadVariableOp2X
*StatsModel/dense_2/MatMul_9/ReadVariableOp*StatsModel/dense_2/MatMul_9/ReadVariableOp2V
)StatsModel/dense_3/BiasAdd/ReadVariableOp)StatsModel/dense_3/BiasAdd/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_1/ReadVariableOp+StatsModel/dense_3/BiasAdd_1/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_2/ReadVariableOp+StatsModel/dense_3/BiasAdd_2/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_3/ReadVariableOp+StatsModel/dense_3/BiasAdd_3/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_4/ReadVariableOp+StatsModel/dense_3/BiasAdd_4/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_5/ReadVariableOp+StatsModel/dense_3/BiasAdd_5/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_6/ReadVariableOp+StatsModel/dense_3/BiasAdd_6/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_7/ReadVariableOp+StatsModel/dense_3/BiasAdd_7/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_8/ReadVariableOp+StatsModel/dense_3/BiasAdd_8/ReadVariableOp2Z
+StatsModel/dense_3/BiasAdd_9/ReadVariableOp+StatsModel/dense_3/BiasAdd_9/ReadVariableOp2T
(StatsModel/dense_3/MatMul/ReadVariableOp(StatsModel/dense_3/MatMul/ReadVariableOp2X
*StatsModel/dense_3/MatMul_1/ReadVariableOp*StatsModel/dense_3/MatMul_1/ReadVariableOp2X
*StatsModel/dense_3/MatMul_2/ReadVariableOp*StatsModel/dense_3/MatMul_2/ReadVariableOp2X
*StatsModel/dense_3/MatMul_3/ReadVariableOp*StatsModel/dense_3/MatMul_3/ReadVariableOp2X
*StatsModel/dense_3/MatMul_4/ReadVariableOp*StatsModel/dense_3/MatMul_4/ReadVariableOp2X
*StatsModel/dense_3/MatMul_5/ReadVariableOp*StatsModel/dense_3/MatMul_5/ReadVariableOp2X
*StatsModel/dense_3/MatMul_6/ReadVariableOp*StatsModel/dense_3/MatMul_6/ReadVariableOp2X
*StatsModel/dense_3/MatMul_7/ReadVariableOp*StatsModel/dense_3/MatMul_7/ReadVariableOp2X
*StatsModel/dense_3/MatMul_8/ReadVariableOp*StatsModel/dense_3/MatMul_8/ReadVariableOp2X
*StatsModel/dense_3/MatMul_9/ReadVariableOp*StatsModel/dense_3/MatMul_9/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:T P
,
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
*__inference_StatsModel_layer_call_fn_10064

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_StatsModel_layer_call_and_return_conditional_losses_90092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_25
serving_default_input_2:0?????????
?;
dense_50
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-1
layer-17
layer_with_weights-2
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"ڄ
_tf_keras_network??{"name": "DualModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "DualModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.unstack_1", "trainable": true, "dtype": "float32", "function": "unstack"}, "name": "tf.unstack_1", "inbound_nodes": [["input_2", 0, 0, {"num": 10, "axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.stack_2", "trainable": true, "dtype": "float32", "function": "stack"}, "name": "tf.stack_2", "inbound_nodes": [[["tf.unstack_1", 0, 0, {"axis": 1}], ["tf.unstack_1", 0, 1, {"axis": 1}], ["tf.unstack_1", 0, 2, {"axis": 1}], ["tf.unstack_1", 0, 3, {"axis": 1}], ["tf.unstack_1", 0, 4, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.stack_3", "trainable": true, "dtype": "float32", "function": "stack"}, "name": "tf.stack_3", "inbound_nodes": [[["tf.unstack_1", 0, 5, {"axis": 1}], ["tf.unstack_1", 0, 6, {"axis": 1}], ["tf.unstack_1", 0, 7, {"axis": 1}], ["tf.unstack_1", 0, 8, {"axis": 1}], ["tf.unstack_1", 0, 9, {"axis": 1}]]]}, {"class_name": "Sequential", "config": {"name": "StatsModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "StatsModel", "inbound_nodes": [[["tf.unstack_1", 0, 0, {}]], [["tf.unstack_1", 0, 1, {}]], [["tf.unstack_1", 0, 2, {}]], [["tf.unstack_1", 0, 3, {}]], [["tf.unstack_1", 0, 4, {}]], [["tf.unstack_1", 0, 5, {}]], [["tf.unstack_1", 0, 6, {}]], [["tf.unstack_1", 0, 7, {}]], [["tf.unstack_1", 0, 8, {}]], [["tf.unstack_1", 0, 9, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.stack_2", 0, 0, {"axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.stack_3", 0, 0, {"axis": 1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "name": "tf.math.reduce_min", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "name": "tf.math.reduce_max", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min_1", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "name": "tf.math.reduce_min_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max_1", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "name": "tf.math.reduce_max_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_3", "inbound_nodes": [[["tf.math.reduce_sum_2", 0, 0, {"axis": 1}], ["tf.math.reduce_sum_3", 0, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["tf.math.reduce_mean", 0, 0, {"axis": 1}], ["tf.math.reduce_min", 0, 0, {"axis": 1}], ["tf.math.reduce_max", 0, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["tf.math.reduce_mean_1", 0, 0, {"axis": 1}], ["tf.math.reduce_min_1", 0, 0, {"axis": 1}], ["tf.math.reduce_max_1", 0, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_4", "inbound_nodes": [[["tf.concat_3", 0, 0, {"axis": 1}], ["tf.concat_1", 0, 0, {"axis": 1}], ["tf.concat_2", 0, 0, {"axis": 1}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["tf.concat_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "shared_object_id": 30, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10, 155]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 155]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 10, 155]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "DualModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "TFOpLambda", "config": {"name": "tf.unstack_1", "trainable": true, "dtype": "float32", "function": "unstack"}, "name": "tf.unstack_1", "inbound_nodes": [["input_2", 0, 0, {"num": 10, "axis": 1}]], "shared_object_id": 1}, {"class_name": "TFOpLambda", "config": {"name": "tf.stack_2", "trainable": true, "dtype": "float32", "function": "stack"}, "name": "tf.stack_2", "inbound_nodes": [[["tf.unstack_1", 0, 0, {"axis": 1}], ["tf.unstack_1", 0, 1, {"axis": 1}], ["tf.unstack_1", 0, 2, {"axis": 1}], ["tf.unstack_1", 0, 3, {"axis": 1}], ["tf.unstack_1", 0, 4, {"axis": 1}]]], "shared_object_id": 2}, {"class_name": "TFOpLambda", "config": {"name": "tf.stack_3", "trainable": true, "dtype": "float32", "function": "stack"}, "name": "tf.stack_3", "inbound_nodes": [[["tf.unstack_1", 0, 5, {"axis": 1}], ["tf.unstack_1", 0, 6, {"axis": 1}], ["tf.unstack_1", 0, 7, {"axis": 1}], ["tf.unstack_1", 0, 8, {"axis": 1}], ["tf.unstack_1", 0, 9, {"axis": 1}]]], "shared_object_id": 3}, {"class_name": "Sequential", "config": {"name": "StatsModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "StatsModel", "inbound_nodes": [[["tf.unstack_1", 0, 0, {}]], [["tf.unstack_1", 0, 1, {}]], [["tf.unstack_1", 0, 2, {}]], [["tf.unstack_1", 0, 3, {}]], [["tf.unstack_1", 0, 4, {}]], [["tf.unstack_1", 0, 5, {}]], [["tf.unstack_1", 0, 6, {}]], [["tf.unstack_1", 0, 7, {}]], [["tf.unstack_1", 0, 8, {}]], [["tf.unstack_1", 0, 9, {}]]], "shared_object_id": 11}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.stack_2", 0, 0, {"axis": 1}]], "shared_object_id": 12}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.stack_3", 0, 0, {"axis": 1}]], "shared_object_id": 13}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 14}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "name": "tf.math.reduce_min", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 15}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "name": "tf.math.reduce_max", "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 16}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 17}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min_1", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "name": "tf.math.reduce_min_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 18}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max_1", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "name": "tf.math.reduce_max_1", "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 19}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_3", "inbound_nodes": [[["tf.math.reduce_sum_2", 0, 0, {"axis": 1}], ["tf.math.reduce_sum_3", 0, 0, {"axis": 1}]]], "shared_object_id": 20}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["tf.math.reduce_mean", 0, 0, {"axis": 1}], ["tf.math.reduce_min", 0, 0, {"axis": 1}], ["tf.math.reduce_max", 0, 0, {"axis": 1}]]], "shared_object_id": 21}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["tf.math.reduce_mean_1", 0, 0, {"axis": 1}], ["tf.math.reduce_min_1", 0, 0, {"axis": 1}], ["tf.math.reduce_max_1", 0, 0, {"axis": 1}]]], "shared_object_id": 22}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_4", "inbound_nodes": [[["tf.concat_3", 0, 0, {"axis": 1}], ["tf.concat_1", 0, 0, {"axis": 1}], ["tf.concat_2", 0, 0, {"axis": 1}]]], "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["tf.concat_4", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 29}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 32}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 5e-05, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 155]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.unstack_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.unstack_1", "trainable": true, "dtype": "float32", "function": "unstack"}, "inbound_nodes": [["input_2", 0, 0, {"num": 10, "axis": 1}]], "shared_object_id": 1}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.stack_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.stack_2", "trainable": true, "dtype": "float32", "function": "stack"}, "inbound_nodes": [[["tf.unstack_1", 0, 0, {"axis": 1}], ["tf.unstack_1", 0, 1, {"axis": 1}], ["tf.unstack_1", 0, 2, {"axis": 1}], ["tf.unstack_1", 0, 3, {"axis": 1}], ["tf.unstack_1", 0, 4, {"axis": 1}]]], "shared_object_id": 2}
?
	keras_api"?
_tf_keras_layer?{"name": "tf.stack_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.stack_3", "trainable": true, "dtype": "float32", "function": "stack"}, "inbound_nodes": [[["tf.unstack_1", 0, 5, {"axis": 1}], ["tf.unstack_1", 0, 6, {"axis": 1}], ["tf.unstack_1", 0, 7, {"axis": 1}], ["tf.unstack_1", 0, 8, {"axis": 1}], ["tf.unstack_1", 0, 9, {"axis": 1}]]], "shared_object_id": 3}
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"name": "StatsModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "StatsModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["tf.unstack_1", 0, 0, {}]], [["tf.unstack_1", 0, 1, {}]], [["tf.unstack_1", 0, 2, {}]], [["tf.unstack_1", 0, 3, {}]], [["tf.unstack_1", 0, 4, {}]], [["tf.unstack_1", 0, 5, {}]], [["tf.unstack_1", 0, 6, {}]], [["tf.unstack_1", 0, 7, {}]], [["tf.unstack_1", 0, 8, {}]], [["tf.unstack_1", 0, 9, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 155}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 155]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 155]}, "float32", "dense_2_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "StatsModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}]}}, "training_config": {"loss": "mae", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
$	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_sum_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.stack_2", 0, 0, {"axis": 1}]], "shared_object_id": 12}
?
%	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_sum_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "inbound_nodes": [["tf.stack_3", 0, 0, {"axis": 1}]], "shared_object_id": 13}
?
&	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 14}
?
'	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_min", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 15}
?
(	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_max", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "inbound_nodes": [[["StatsModel", 1, 0, {"axis": 0}], ["StatsModel", 2, 0, {"axis": 0}], ["StatsModel", 3, 0, {"axis": 0}], ["StatsModel", 4, 0, {"axis": 0}], ["StatsModel", 5, 0, {"axis": 0}]]], "shared_object_id": 16}
?
)	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_mean_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_1", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 17}
?
*	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_min_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_min_1", "trainable": true, "dtype": "float32", "function": "math.reduce_min"}, "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 18}
?
+	keras_api"?
_tf_keras_layer?{"name": "tf.math.reduce_max_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_max_1", "trainable": true, "dtype": "float32", "function": "math.reduce_max"}, "inbound_nodes": [[["StatsModel", 6, 0, {"axis": 0}], ["StatsModel", 7, 0, {"axis": 0}], ["StatsModel", 8, 0, {"axis": 0}], ["StatsModel", 9, 0, {"axis": 0}], ["StatsModel", 10, 0, {"axis": 0}]]], "shared_object_id": 19}
?
,	keras_api"?
_tf_keras_layer?{"name": "tf.concat_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}, "inbound_nodes": [[["tf.math.reduce_sum_2", 0, 0, {"axis": 1}], ["tf.math.reduce_sum_3", 0, 0, {"axis": 1}]]], "shared_object_id": 20}
?
-	keras_api"?
_tf_keras_layer?{"name": "tf.concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "inbound_nodes": [[["tf.math.reduce_mean", 0, 0, {"axis": 1}], ["tf.math.reduce_min", 0, 0, {"axis": 1}], ["tf.math.reduce_max", 0, 0, {"axis": 1}]]], "shared_object_id": 21}
?
.	keras_api"?
_tf_keras_layer?{"name": "tf.concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "inbound_nodes": [[["tf.math.reduce_mean_1", 0, 0, {"axis": 1}], ["tf.math.reduce_min_1", 0, 0, {"axis": 1}], ["tf.math.reduce_max_1", 0, 0, {"axis": 1}]]], "shared_object_id": 22}
?
/	keras_api"?
_tf_keras_layer?{"name": "tf.concat_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}, "inbound_nodes": [[["tf.concat_3", 0, 0, {"axis": 1}], ["tf.concat_1", 0, 0, {"axis": 1}], ["tf.concat_2", 0, 0, {"axis": 1}]]], "shared_object_id": 23}
?	

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.concat_4", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 442}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 442]}}
?	

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?0mq1mr6ms7mt<mu=mv>mw?mx0vy1vz6v{7v|<v}=v~>v?v?"
	optimizer
 "
trackable_list_wrapper
X
<0
=1
>2
?3
04
15
66
77"
trackable_list_wrapper
X
<0
=1
>2
?3
04
15
66
77"
trackable_list_wrapper
?
@non_trainable_variables
regularization_losses

Alayers
Blayer_regularization_losses
Cmetrics
Dlayer_metrics
	variables
trainable_variables
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
_generic_user_object
"
_generic_user_object
?	

<kernel
=bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 155]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 155}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 155]}}
?

>kernel
?bias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
c<m?=m?>m??m?<v?=v?>v??v?"
	optimizer
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
?
Mnon_trainable_variables
 regularization_losses

Nlayers
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
!	variables
"trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
!:	?d2dense_4/kernel
:d2dense_4/bias
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
Rnon_trainable_variables
2regularization_losses

Slayers
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
3	variables
4trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :d2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
Wnon_trainable_variables
8regularization_losses

Xlayers
Ylayer_regularization_losses
Zmetrics
[layer_metrics
9	variables
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
!:	?2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
?
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
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
^non_trainable_variables
Eregularization_losses

_layers
`layer_regularization_losses
ametrics
blayer_metrics
F	variables
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
cnon_trainable_variables
Iregularization_losses

dlayers
elayer_regularization_losses
fmetrics
glayer_metrics
J	variables
Ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
?
	htotal
	icount
j	variables
k	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 37}
?
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 32}
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
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
!:	?d2dense_4/kernel/m
:d2dense_4/bias/m
 :d2dense_5/kernel/m
:2dense_5/bias/m
": 
??2dense_2/kernel/m
:?2dense_2/bias/m
!:	?2dense_3/kernel/m
:2dense_3/bias/m
!:	?d2dense_4/kernel/v
:d2dense_4/bias/v
 :d2dense_5/kernel/v
:2dense_5/bias/v
": 
??2dense_2/kernel/v
:?2dense_2/bias/v
!:	?2dense_3/kernel/v
:2dense_3/bias/v
": 
??2dense_2/kernel/m
:?2dense_2/bias/m
!:	?2dense_3/kernel/m
:2dense_3/bias/m
": 
??2dense_2/kernel/v
:?2dense_2/bias/v
!:	?2dense_3/kernel/v
:2dense_3/bias/v
?2?
__inference__wrapped_model_8908?
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
input_2?????????
?
?2?
C__inference_DualModel_layer_call_and_return_conditional_losses_9808
C__inference_DualModel_layer_call_and_return_conditional_losses_9962
C__inference_DualModel_layer_call_and_return_conditional_losses_9521
C__inference_DualModel_layer_call_and_return_conditional_losses_9631?
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
(__inference_DualModel_layer_call_fn_9218
(__inference_DualModel_layer_call_fn_9983
)__inference_DualModel_layer_call_fn_10004
(__inference_DualModel_layer_call_fn_9411?
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
?2?
E__inference_StatsModel_layer_call_and_return_conditional_losses_10021
E__inference_StatsModel_layer_call_and_return_conditional_losses_10038
D__inference_StatsModel_layer_call_and_return_conditional_losses_9047
D__inference_StatsModel_layer_call_and_return_conditional_losses_9061?
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
)__inference_StatsModel_layer_call_fn_8960
*__inference_StatsModel_layer_call_fn_10051
*__inference_StatsModel_layer_call_fn_10064
)__inference_StatsModel_layer_call_fn_9033?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_10075?
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
'__inference_dense_4_layer_call_fn_10084?
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
B__inference_dense_5_layer_call_and_return_conditional_losses_10095?
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
'__inference_dense_5_layer_call_fn_10104?
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
"__inference_signature_wrapper_9654input_2"?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_10115?
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
'__inference_dense_2_layer_call_fn_10124?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_10134?
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
'__inference_dense_3_layer_call_fn_10143?
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
 ?
C__inference_DualModel_layer_call_and_return_conditional_losses_9521p<=>?0167=?:
3?0
&?#
input_2?????????
?
p 

 
? "%?"
?
0?????????
? ?
C__inference_DualModel_layer_call_and_return_conditional_losses_9631p<=>?0167=?:
3?0
&?#
input_2?????????
?
p

 
? "%?"
?
0?????????
? ?
C__inference_DualModel_layer_call_and_return_conditional_losses_9808o<=>?0167<?9
2?/
%?"
inputs?????????
?
p 

 
? "%?"
?
0?????????
? ?
C__inference_DualModel_layer_call_and_return_conditional_losses_9962o<=>?0167<?9
2?/
%?"
inputs?????????
?
p

 
? "%?"
?
0?????????
? ?
)__inference_DualModel_layer_call_fn_10004b<=>?0167<?9
2?/
%?"
inputs?????????
?
p

 
? "???????????
(__inference_DualModel_layer_call_fn_9218c<=>?0167=?:
3?0
&?#
input_2?????????
?
p 

 
? "???????????
(__inference_DualModel_layer_call_fn_9411c<=>?0167=?:
3?0
&?#
input_2?????????
?
p

 
? "???????????
(__inference_DualModel_layer_call_fn_9983b<=>?0167<?9
2?/
%?"
inputs?????????
?
p 

 
? "???????????
E__inference_StatsModel_layer_call_and_return_conditional_losses_10021g<=>?8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_StatsModel_layer_call_and_return_conditional_losses_10038g<=>?8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_StatsModel_layer_call_and_return_conditional_losses_9047n<=>???<
5?2
(?%
dense_2_input??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_StatsModel_layer_call_and_return_conditional_losses_9061n<=>???<
5?2
(?%
dense_2_input??????????
p

 
? "%?"
?
0?????????
? ?
*__inference_StatsModel_layer_call_fn_10051Z<=>?8?5
.?+
!?
inputs??????????
p 

 
? "???????????
*__inference_StatsModel_layer_call_fn_10064Z<=>?8?5
.?+
!?
inputs??????????
p

 
? "???????????
)__inference_StatsModel_layer_call_fn_8960a<=>???<
5?2
(?%
dense_2_input??????????
p 

 
? "???????????
)__inference_StatsModel_layer_call_fn_9033a<=>???<
5?2
(?%
dense_2_input??????????
p

 
? "???????????
__inference__wrapped_model_8908t<=>?01675?2
+?(
&?#
input_2?????????
?
? "1?.
,
dense_5!?
dense_5??????????
B__inference_dense_2_layer_call_and_return_conditional_losses_10115^<=0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_2_layer_call_fn_10124Q<=0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_3_layer_call_and_return_conditional_losses_10134]>?0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_10143P>?0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_4_layer_call_and_return_conditional_losses_10075]010?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? {
'__inference_dense_4_layer_call_fn_10084P010?-
&?#
!?
inputs??????????
? "??????????d?
B__inference_dense_5_layer_call_and_return_conditional_losses_10095\67/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? z
'__inference_dense_5_layer_call_fn_10104O67/?,
%?"
 ?
inputs?????????d
? "???????????
"__inference_signature_wrapper_9654<=>?0167@?=
? 
6?3
1
input_2&?#
input_2?????????
?"1?.
,
dense_5!?
dense_5?????????