"""base_dataset + motion 的冒烟测试。

单独一个文件，不动 base_dataset.py 一行——这样 base_dataset 保持官方原版的 151 行，
你读代码时不会被 200 行断言淹没。

    python prepare_data/smoke_test.py                 # 官方原版直接跑
    python prepare_data/smoke_test.py --mode view     # 带 7 个视角

分两段：
  阶段 0/1  用官方代码原样构造 Dataset，失败则打印「官方假设了什么 / 实际是什么」
  阶段 2-8  构造成功后逐条验证 train 路径上的功能
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from functional.motion import (  # noqa: E402
    normalize_motion,
    trans_motion3d,
    trans_motion_inv,
)
from prepare_data.base_dataset import _MixamoDatasetBase  # noqa: E402

# 与官方 common.py 一致
DEFAULT_VIEW_ANGLES = [(0, 0, -np.pi / 2),
                       (0, 0, -np.pi / 3),
                       (0, 0, -np.pi / 6),
                       (0, 0, 0),
                       (0, 0, np.pi / 6),
                       (0, 0, np.pi / 3),
                       (0, 0, np.pi / 2)]

# 官方 hardcode 的两套角色名，用来演示对不上的地方
OFFICIAL_TRAIN_CHARS = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm',
                        'Pearl', 'Warrok', 'Globin', 'Kaya', 'PeanutMan']
OFFICIAL_TEST_CHARS = ['Ty', 'Andromeda', 'Pumpkinhulk', 'SportyGranny']


class _ProbeConfig(object):
    """替身 config，字段与官方 common.py 对齐。"""

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.unit = args.unit
        self.view_angles = None if args.mode == 'skeleton' else list(DEFAULT_VIEW_ANGLES)
        suffix = '' if args.mode == 'skeleton' else '_with_view'
        self.meanpose_path = os.path.join(args.cache_dir, 'meanpose%s.npy' % suffix)
        self.stdpose_path = os.path.join(args.cache_dir, 'stdpose%s.npy' % suffix)


_RESULTS = []


def _check(tag, cond, detail=''):
    _RESULTS.append((tag, bool(cond)))
    print('      [%s] %-46s %s' % ('PASS' if cond else 'FAIL', tag, detail))
    return bool(cond)


def _hr(ch='='):
    print(ch * 74)


# ---------------------------------------------------------------- 布局诊断

def _detect_layout(root):
    import glob
    if not os.path.isdir(root):
        return None
    if glob.glob(os.path.join(root, '*', '*', 'motions', '*.npy')):
        return 'nested'
    if glob.glob(os.path.join(root, '*__*', '*.npy')):
        return 'flat'
    return None


def preflight(cfg):
    """把官方代码的三个隐含假设逐条摊开，看数据实际满足几条。"""
    _hr()
    print(' 阶段 0  官方代码隐含的三个前提')
    _hr()

    phase_root = os.path.join(cfg.data_dir, 'train')
    ok_dir = os.path.isdir(phase_root)
    print('  1) 数据目录要按 phase 分层：<data_dir>/train、<data_dir>/test')
    print('     期望 %s' % phase_root)
    print('     实际 %s' % ('存在' if ok_dir else '不存在'))
    print('     绕过办法：把全部数据直接放在 data_dir 下（现在就是），但官方代码不会去那儿找')

    prot = OFFICIAL_TRAIN_CHARS[0]          # 'Aj'
    nested = os.path.join(phase_root, prot, 'Golf Drive', 'motions')
    print('  2) 角色名要叫官方那 15 个：%s ...' % ', '.join(OFFICIAL_TRAIN_CHARS[:4]))
    print('     它去找 %s' % nested)
    print('     实际 %s' % ('存在' if os.path.isdir(nested) else '不存在'))

    layout = _detect_layout(cfg.data_dir)
    print('  3) 目录要三层：<char>/<anim>/motions/<i>.npy')
    print('     实际根目录 %s 的布局 = %s' % (cfg.data_dir, layout))
    if layout == 'flat':
        dirs = [d for d in os.listdir(cfg.data_dir)
                if os.path.isdir(os.path.join(cfg.data_dir, d)) and '__' in d]
        chars = sorted({d.split('__', 1)[0] for d in dirs})
        print('     实际角色名 %d 个：%s' % (len(chars), ', '.join(chars[:6]) + ' ...'))
    print()


def try_build(cfg):
    """用官方代码原样构造，失败就打印原因。"""
    _hr()
    print(' 阶段 1  用官方代码原样构造 Dataset')
    _hr()
    t0 = time.time()
    try:
        ds = _MixamoDatasetBase('train', cfg)
    except Exception as e:
        print('  构造失败：%s: %s' % (type(e).__name__, e))
        print()
        print('  traceback 末尾：')
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            print('    ' + line)
        print()
        print('  直接原因：character_names[0] = %r 在数据里不存在，' % OFFICIAL_TRAIN_CHARS[0])
        print('            glob 返回空列表 -> motion_names = []，')
        print('            继续走到 gen_meanpose 的 np.concatenate([]) 才炸。')
        return None
    print('  构造成功  用时 %.2fs' % (time.time() - t0))
    return ds


# ---------------------------------------------------------------- 功能断言

def _skeleton_item(ds):
    """复刻 MixamoDatasetForSkeleton.__getitem__ 的组装逻辑。"""
    idx1, idx2 = np.random.choice(len(ds.motion_names), size=2, replace=False)
    mot1, mot2 = ds.motion_names[idx1], ds.motion_names[idx2]
    idx1, idx2 = np.random.choice(len(ds.character_names), size=2, replace=False)
    char1, char2 = ds.character_names[idx1], ds.character_names[idx2]

    if ds.aug:
        p1 = ds.gen_aug_param(rotate=True)
        p2 = ds.gen_aug_param(rotate=True)
        p12 = {'ratio': p2['ratio'], 'roll': p1['roll']}
        p21 = {'ratio': p1['ratio'], 'roll': p2['roll']}
    else:
        p1 = p2 = p12 = p21 = None

    return {
        'input1': ds.preprocessing(ds.build_item(mot1, char1), param=p1),
        'input2': ds.preprocessing(ds.build_item(mot2, char2), param=p2),
        'input12': ds.preprocessing(ds.build_item(mot1, char2), param=p12),
        'input21': ds.preprocessing(ds.build_item(mot2, char1), param=p21),
        'mot1': mot1, 'mot2': mot2, 'char1': char1, 'char2': char2,
    }


def run_checks(ds, cfg, args):
    t0 = time.time()

    _hr()
    print(' 阶段 2-8  功能断言')
    _hr()

    print('\n-- 2 索引 / build_item')
    print('      layout     = %s' % getattr(ds, 'layout', '(官方版无此属性)'))
    print('      data_root  = %s' % ds.data_root)
    print('      characters = %d  %s' % (len(ds.character_names), ' '.join(ds.character_names)))
    print('      motions    = %d' % len(ds.motion_names))
    print('      __len__    = %d' % len(ds))
    _check('角色数 / 动作数均 > 1',
           len(ds.character_names) > 1 and len(ds.motion_names) > 1,
           '%d x %d' % (len(ds.character_names), len(ds.motion_names)))

    if not ds.motion_names:
        print()
        _hr('!')
        print(' motion_names 为空，后面全部无意义 —— 这是官方最危险的一种失败：')
        print(' __init__ 没有抛异常（因为 mean/std 缓存命中，跳过了 gen_meanpose），')
        print(' 你以为构建成功了，实际 __len__ = 0，DataLoader 会在第一次迭代时报')
        print(' "num_samples should be a positive integer"。')
        print(' 把 %s 删掉再跑一次，就能看到它真正崩在 np.concatenate([])。' % cfg.meanpose_path)
        _hr('!')
        return 1

    missing = [ds.build_item(m, c) for c in ds.character_names for m in ds.motion_names
               if not os.path.exists(ds.build_item(m, c))]
    _check('每个 (char, motion) 都有文件', not missing,
           '缺 %d 个' % len(missing) if missing else '%d 全在' % len(ds))

    print('\n-- 3 mean/std')
    print('      mean_pose %s   std_pose %s' % (ds.mean_pose.shape, ds.std_pose.shape))
    _check('形状为 (15, 2)', ds.mean_pose.shape == (15, 2) and ds.std_pose.shape == (15, 2))
    _check('std 无 0', np.all(ds.std_pose > 0), 'min=%.4f' % ds.std_pose.min())
    _check('缓存已落盘',
           os.path.exists(cfg.meanpose_path) and os.path.exists(cfg.stdpose_path))

    print('\n-- 4 单窗口 preprocessing')
    ds.aug = False
    item = ds.build_item(ds.motion_names[0], ds.character_names[0])
    raw = np.load(item)
    out = ds.preprocessing(item, None, None)
    print('      %s -> %s %s' % (tuple(raw.shape), tuple(out.shape), out.dtype))
    _check('输出 (30, 64)', tuple(out.shape) == (30, 64))
    _check('dtype = float32', out.dtype == torch.float32, str(out.dtype))
    _check('无 NaN/Inf', bool(torch.isfinite(out).all()))
    _check('不是常数', float(out.std()) > 1e-3, 'std=%.4f' % float(out.std()))

    print('\n-- 5 与手算逐步对拍')
    u = float(cfg.unit)
    ref = raw.astype(np.float64) * u
    ref[1] = (ref[2] + ref[5]) / 2
    ref[8] = (ref[9] + ref[12]) / 2
    proj = ref[:, [0, 2], :].copy()
    proj[:, 1, :] *= -1
    centers = proj[8].copy()
    vel = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    packed = np.r_[(proj - centers)[:8], (proj - centers)[9:], vel]
    expect = ((packed - ds.mean_pose[:, :, None]) / ds.std_pose[:, :, None]).reshape(-1, 64)
    err = float(np.abs(expect - out.numpy()).max())
    _check('逐元素最大误差 < 1e-4', err < 1e-4, 'max|diff| = %.3e' % err)

    print('\n-- 6 可逆性')
    r = raw.astype(np.float64) * u
    r[1] = (r[2] + r[5]) / 2
    r[8] = (r[9] + r[12]) / 2
    proj_before = r[:, [0, 2], :].copy()
    proj_before[:, 1, :] *= -1
    proj_after = trans_motion3d(raw.astype(np.float64), None, cfg.unit)
    inv = trans_motion_inv(proj_after, sx=0, sy=0)
    base = proj_before[8][:, :1][None, :, :]
    err_inv = float(np.abs(inv + base - proj_before).max())
    _check('trans_motion_inv 还原误差 < 1e-5', err_inv < 1e-5, 'max|diff| = %.3e' % err_inv)
    _check('根速度第 0 帧为 0', float(np.abs(proj_after[-1, :, 0]).max()) < 1e-9)

    print('\n-- 7 增广')
    ds.aug = True
    p = ds.gen_aug_param(rotate=True)
    _check('含 ratio 与 roll', 'ratio' in p and 'roll' in p and len(p['roll']) == 3)
    _check('ratio ∈ [0.8, 1.2]（注意官方 vs 论文 (0.5,1.5)）',
           0.8 <= p['ratio'] <= 1.2, 'ratio=%.4f' % p['ratio'])
    a = ds.preprocessing(item, None, p)
    b = ds.preprocessing(item, None, p)
    _check('同一 param 两次一致', bool(torch.allclose(a, b)))
    c = ds.preprocessing(item, None, ds.gen_aug_param(rotate=True))
    _check('不同 param 结果不同', not bool(torch.allclose(a, c)))
    _check('增广后 dtype 仍为 float32', a.dtype == torch.float32, str(a.dtype))

    print('\n-- 8 交叉组装 / collate')
    ds.aug = False
    ok_shape = ok_self = ok_cross = ok_diff = True
    for _ in range(args.iters):
        d = _skeleton_item(ds)
        for k in ('input1', 'input2', 'input12', 'input21'):
            ok_shape &= tuple(d[k].shape) == (30, 64)
        ok_self &= bool(torch.allclose(
            d['input1'], ds.preprocessing(ds.build_item(d['mot1'], d['char1']))))
        ok_cross &= bool(torch.allclose(
            d['input12'], ds.preprocessing(ds.build_item(d['mot1'], d['char2']))))
        ok_diff &= not bool(torch.allclose(d['input1'], d['input12']))
    _check('四个张量均 (30, 64)', ok_shape, '%d 轮抽样' % args.iters)
    _check('input1 == preprocess(mot1, char1)', ok_self)
    _check('input12 == preprocess(mot1, char2)', ok_cross)
    _check('input1 != input12', ok_diff)

    ds.aug = True
    batch = torch.utils.data.default_collate([_skeleton_item(ds) for _ in range(args.batch_size)])
    print('      input1 -> %s' % (tuple(batch['input1'].shape),))
    _check('batch 形状 (B, 30, 64)', tuple(batch['input1'].shape) == (args.batch_size, 30, 64))
    _check('batch dtype float32', batch['input1'].dtype == torch.float32)

    if args.mode != 'skeleton':
        print('\n-- 附加 视角投影')
        ds.aug = False
        outs = [ds.preprocessing(item, v, None) for v in cfg.view_angles]
        _check('7 视角均 (30,64) 且有限',
               all(tuple(o.shape) == (30, 64) for o in outs)
               and all(bool(torch.isfinite(o).all()) for o in outs))
        _check('不同视角互不相同', all(not torch.allclose(outs[0], o) for o in outs[1:]))
        vert = all(torch.allclose(outs[0].reshape(15, 2, 64)[:, 1],
                                  o.reshape(15, 2, 64)[:, 1], atol=1e-5) for o in outs[1:])
        _check('竖直分量跨视角恒定', vert)

    # 汇总
    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print()
    _hr('-')
    print(' %d/%d 通过   用时 %.2fs' % (passed, total, time.time() - t0))
    if passed != total:
        print(' 未通过：')
        for tag, ok in _RESULTS:
            if not ok:
                print('   - %s' % tag)
    _hr('-')
    return 0 if passed == total else 1


def main():
    ap = argparse.ArgumentParser(description='base_dataset + motion 冒烟测试')
    ap.add_argument('--data-dir', default=os.path.join(_ROOT, 'data', 'motion'))
    ap.add_argument('--cache-dir', default=os.path.join(_ROOT, 'data'))
    ap.add_argument('--mode', default='skeleton', choices=['skeleton', 'view', 'full'])
    ap.add_argument('--unit', type=int, default=128)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    cfg = _ProbeConfig(args)
    _hr()
    print(' base_dataset 冒烟测试')
    print(' data_dir = %s' % os.path.abspath(cfg.data_dir))
    print(' mode     = %s   phase = train   unit = %d' % (args.mode, cfg.unit))
    _hr()

    preflight(cfg)
    ds = try_build(cfg)
    if ds is None:
        print(' 阶段 2-8 跳过：先把上面的前提补上。')
        return 1
    return run_checks(ds, cfg, args)


if __name__ == '__main__':
    sys.exit(main())
