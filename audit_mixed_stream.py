"""Audit-only mixed GRU/LSTM export, reusing export_v4's validated cells.
No zip or submission: three prefixed single-model graphs share raw/previous-price
inputs; each preserves its own explicit standardization. Raw heads averaged once.
"""
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import torch
import export_v4


def export(ckpts, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    checkpoints=[torch.load(path,map_location='cpu',weights_only=False) for path in ckpts]
    for ck in checkpoints:
        assert ck['step']==64500
        for key in ('price_cols','other_cols','flag_cols'):assert ck[key]==checkpoints[0][key]
    nodes=[];initializers=[];states_in=[];states_out=[];heads=[];price_idx=None
    for i,ck in enumerate(checkpoints):
        m,stats=export_v4.build(ck,cell='matmul',fold=False)
        prefix=f'm{i}/';m=onnx.compose.add_prefix(m,prefix)
        mapping={prefix+'state':'state',prefix+'prev_prices':'prev_prices'}
        if i==0:mapping[prefix+'prices']='prices'
        for node in m.graph.node:
            for j,name in enumerate(node.input):node.input[j]=mapping.get(name,name)
            for j,name in enumerate(node.output):node.output[j]=mapping.get(name,name)
            nodes.append(node)
        initializers.extend(m.graph.initializer)
        for item in m.graph.input:
            if item.name not in (prefix+'state',prefix+'prev_prices'):states_in.append(item)
        for item in m.graph.output:
            if item.name.endswith('_out'):states_out.append(item)
        heads.append(prefix+'pred');price_idx=stats['price_idx']
    nodes+=[h.make_node('Add',heads[:2],['sum01']),h.make_node('Add',['sum01',heads[2]],['sum012']),
            h.make_node('Div',['sum012','count'],['average']),h.make_node('Clip',['average','lo','hi'],['pred'])]
    initializers += [nh.from_array(np.float32(3),'count'),nh.from_array(np.float32(-2),'lo'),nh.from_array(np.float32(2),'hi')]
    info=lambda name,shape:h.make_tensor_value_info(name,T.FLOAT,shape)
    graph=h.make_graph(nodes,'fixed_two_gru_one_lstm_audit',
                       [info('state',[1,1,112]),info('prev_prices',[1,1,44])]+states_in,
                       [info('pred',[1,1,2])]+states_out+[info('prices',[1,1,44])],initializers)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=8;onnx.checker.check_model(model,full_check=True)
    from onnxruntime.quantization import quantize_dynamic,QuantType
    for kind in ('fp32','int8'):
        directory=out/kind;directory.mkdir(exist_ok=False)
        if kind=='fp32':onnx.save(model,directory/'model.onnx')
        else:quantize_dynamic(str(out/'fp32/model.onnx'),str(directory/'model.onnx'),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul'])
        np.savez(directory/'stats.npz',price_idx=price_idx)
        shutil.copyfile(Path(__file__).parent/'solution_v4.py',directory/'solution.py')
    return checkpoints
