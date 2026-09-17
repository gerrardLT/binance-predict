"""绘制 15m 孕线倒垂线反转信号示意图"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_hanging_man():
    """绘制上吊线（Hanging Man） - 下跌趋势中的卖出信号"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== 左上：上吊线形态（下跌趋势后）==========
    ax1 = axes[0, 0]
    ax1.set_title('(A) 上吊线 (Hanging Man)\n下跌趋势后的看跌反转信号', fontsize=12, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)
    
    x = [0, 1, 2, 3]  # 当前这根是第 4 根
    
    # 前 3 根大阴线（下跌趋势）
    candles = [
        {'o': 100, 'h': 98, 'l': 102, 'c': 97},   # 第 0 根
        {'o': 98, 'h': 100, 'l': 96, 'c': 99},   # 第 1 根  
        {'o': 97, 'h': 95, 'l': 99, 'c': 94},    # 第 2 根
    ]
    
    atr20 = 2.0
    
    for i, candle in enumerate(candles):
        o, h, l, c = candle['o'], candle['h'], candle['l'], candle['c']
        color = '#E53935' if c < o else '#43A047'  # 红涨绿跌（中国习惯）
        
        # 实体
        ax1.plot([i, i], [max(o,c), min(o,c)], color=color, linewidth=5)
        # 影线
        ax1.plot([i, i], [h, l], color=color, linewidth=1.5)
    
    # 第 4 根：上吊线信号棒
    signal_o = 94.0
    signal_c = 93.5  # 弱收盘
    signal_h = 94.3   # 上影极短≤0.15×ATR = 0.3
    signal_l = 91.5    # 下影很长≥2×实体且≥0.3×ATR = 4.0
    
    body_height = abs(signal_o - signal_c)  # 0.5
    lower_shadow = min(signal_o, signal_c) - signal_l  # 2.5
    upper_shadow = signal_h - max(signal_o, signal_c)  # 0.3
    
    sig_x = 3
    color = '#43A047'  # 阳线（但这是看跌信号！）
    
    ax1.plot([sig_x, sig_x], [max(signal_o, signal_c), min(signal_o, signal_c)], 
             color=color, linewidth=5, label='小实体 (≤0.3×ATR)', zorder=5)
    ax1.plot([sig_x, sig_x], [signal_h, max(signal_o, signal_c)], color=color, linewidth=1.5, zorder=5)
    ax1.plot([sig_x, sig_x], [min(signal_o, signal_c), signal_l], color=color, linewidth=1.5, zorder=5)
    
    # 标注
    ax1.annotate(f'上影 = {upper_shadow:.1f}', 
                 xy=(sig_x, (signal_h + max(signal_o, signal_c))/2),
                 xytext=(60, -40), arrowprops=dict(arrowstyle='->', color='purple'),
                 fontsize=10, color='purple', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCCBC'))
    ax1.annotate(f'下影 = {lower_shadow:.1f}', 
                 xy=(sig_x, (signal_l + min(signal_o, signal_c))/2),
                 xytext=(-100, -60), arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=10, color='blue', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#BBDEFB'))
    ax1.annotate(f'实体 = {body_height:.1f}', 
                 xy=(sig_x, (min(signal_o, signal_c) + max(signal_o, signal_c))/2),
                 xytext=(-70, 30), arrowprops=dict(arrowstyle='->', color='darkgreen'),
                 fontsize=10, color='darkgreen', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9'))
    
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(90, 103)
    
    # ========== 右上：倒锤头形态（上涨趋势后）==========
    ax2 = axes[0, 1]
    ax2.set_title('(B) 倒锤头 (Inverted Hammer)\n上涨趋势后的看跌反转信号', fontsize=12, fontweight='bold')
    ax2.set_ylabel('价格')
    ax2.grid(True, alpha=0.3)
    
    candles_up = [
        {'o': 100, 'h': 102, 'l': 98, 'c': 103},   # 第 0 根
        {'o': 103, 'h': 105, 'l': 101, 'c': 104},  # 第 1 根
        {'o': 104, 'h': 106, 'l': 103, 'c': 105},  # 第 2 根
    ]
    
    for i, candle in enumerate(candles_up):
        o, h, l, c = candle['o'], candle['h'], candle['l'], candle['c']
        color = '#E53935' if c < o else '#43A047'
        ax2.plot([i, i], [max(o,c), min(o,c)], color=color, linewidth=5)
        ax2.plot([i, i], [h, l], color=color, linewidth=1.5)
    
    # 倒锤头信号棒
    ih_o = 105.0
    ih_c = 105.8
    ih_h = 108.5    # 上影很长
    ih_l = 104.8     # 下影极短
    
    ih_body = abs(ih_o - ih_c)
    ih_upper = ih_h - max(ih_o, ih_c)
    ih_lower = min(ih_o, ih_c) - ih_l
    
    ih_x = 3
    color = '#E53935'
    
    ax2.plot([ih_x, ih_x], [max(ih_o, ih_c), min(ih_o, ih_c)], 
             color=color, linewidth=5, label='小实体 (≤0.3×ATR)', zorder=5)
    ax2.plot([ih_x, ih_x], [ih_h, max(ih_o, ih_c)], color=color, linewidth=1.5, zorder=5)
    ax2.plot([ih_x, ih_x], [min(ih_o, ih_c), ih_l], color=color, linewidth=1.5, zorder=5)
    
    ax2.annotate(f'上影 = {ih_upper:.1f}', 
                 xy=(ih_x, (ih_h + max(ih_o, ih_c))/2),
                 xytext=(60, 30), arrowprops=dict(arrowstyle='->', color='orange'),
                 fontsize=10, color='orange', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE0B2'))
    ax2.annotate(f'下影 ≈ 0', 
                 xy=(ih_x, (ih_l + min(ih_o, ih_c))/2),
                 xytext=(-50, -30), arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0'))
    ax2.annotate(f'实体 = {ih_body:.1f}', 
                 xy=(ih_x, (min(ih_o, ih_c) + max(ih_o, ih_c))/2),
                 xytext=(-60, -40), arrowprops=dict(arrowstyle='->', color='darkgreen'),
                 fontsize=10, color='darkgreen', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9'))
    
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(103, 110)
    
    # ========== 左下：触发条件清单 ==========
    ax3 = axes[1, 0]
    ax3.set_title('📐 触发条件（15m 收盘判定·几何阈值×ATR20）', fontsize=11, fontweight='bold', pad=20)
    ax3.axis('off')
    
    conditions = [
        "① 实体 ≤ 0.3×ATR20",
        "② 下影 ≥ 2×实体 且 下影 ≥ 0.3×ATR20", 
        "③ 上影 ≤ 0.15×ATR20（极短或无上影）",
        "④ 收盘距近 20 根最高 ≤ 0.75×ATR20",
        "⑤ CLV = (C-L)/(H-L) ≤ 0.75（弱收盘靠下）",
        "⑥ K 线连续（无缺棒、无跳空）",
    ]
    
    ax3.text(0.05, 0.95, '\n'.join(conditions), fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9, edgecolor='#FFB74D'))
    
    # ========== 右下：入场与结算规则 ==========
    ax4 = axes[1, 1]
    ax4.set_title('🎯 入场规则（目标周期 15m）', fontsize=11, fontweight='bold', pad=20)
    ax4.axis('off')
    
    entry_rules = [
        "障碍位：O ± 0.25×ATR（锚点 O=目标周期开盘价）",
        "",
        "触及触发：",
        "• mid ≥ O+0.25×ATR 且发生在 600s 内",
        "→ 押 DOWN（预期收阴）",
        "",
        "放弃情形：",
        "• 先破 O-0.25×ATR → ABANDON_LOWER",
        "• 触及在 600s 后 → ABANDON_LATE",
        "",
        "结算条件：",
        "• TOUCHED 状态 + 目标根 close < open = WIN",
        "• 其他情况均为 LOSS/EXPIRED/NOISE"
    ]
    
    ax4.text(0.05, 0.95, '\n'.join(entry_rules), fontsize=10.5,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9, edgecolor='#66BB6A')))
    
    plt.suptitle('15m 孕线倒垂线反转信号详解\nHM Touch Down v1/v2 (Hanging Man / Inverted Hammer)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = r'd:\project\binance-predict\docs\hanging_man_inverted_hammer_example.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ K 线示意图已保存：{output_path}")
    
    return output_path

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    draw_hanging_man()
