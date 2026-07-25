/**
 * FlowER - Flower Embodied Robotics
 * 全局交互脚本
 */

'use strict';

// ============ API Client ============
const FlowerAPI = {
    async get(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('API GET Error:', error);
            showToast('网络请求失败', 'error');
            return null;
        }
    },

    async post(url, data = {}) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('API POST Error:', error);
            showToast('操作失败，请重试', 'error');
            return null;
        }
    }
};

// ============ Toast Notification ============
function showToast(message, type = 'success', duration = 3000) {
    // 创建 toast 元素
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'error' : ''}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // 点击手动关闭
    toast.addEventListener('click', () => {
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    });

    // 自动移除
    setTimeout(() => {
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============ Page Initialization ============
document.addEventListener('DOMContentLoaded', () => {
    // 页面淡入动画
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease';
    requestAnimationFrame(() => {
        document.body.style.opacity = '1';
    });

    // 导航高亮当前页面
    highlightCurrentNav();

    // 初始化页面特定功能
    initDiaryPage();

    // 移动端菜单适配
    initMobileNav();
});

// ============ Navigation Highlight ============
function highlightCurrentNav() {
    const path = decodeURIComponent(window.location.pathname);
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        const href = decodeURIComponent(link.getAttribute('href'));
        if (path === href || (path.startsWith(href) && href !== '/')) {
            link.style.color = 'var(--primary-dark)';
            link.style.background = 'var(--primary-light)';
            link.style.fontWeight = '600';
        }
    });
}

// ============ Mobile Navigation ============
function initMobileNav() {
    const navLinks = document.querySelector('.nav-links');
    if (!navLinks) return;

    // 窄屏时为导航添加可切换展开行为
    const logo = document.querySelector('.nav-logo');
    if (!logo) return;

    // 仅在窄屏时激活
    if (window.innerWidth > 600) return;

    navLinks.style.display = 'none';
    let expanded = false;

    logo.addEventListener('click', (e) => {
        // 仅在窄屏模式下拦截
        if (window.innerWidth > 600) return;
        e.preventDefault();
        expanded = !expanded;
        navLinks.style.display = expanded ? 'flex' : 'none';
    });

    // 点击链接后自动收起
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 600) {
                expanded = false;
                navLinks.style.display = 'none';
            }
        });
    });
}

// ============ Diary Page Functions ============
function initDiaryPage() {
    const nowCardsContainer = document.getElementById('now-cards');
    if (!nowCardsContainer) return; // 不在日记页面

    // 添加手动生成按钮
    addGenerateButton(nowCardsContainer);

    // 设置定时生成（演示模式：每60秒）
    setInterval(generateNewCard, 60000);
}

function addGenerateButton(container) {
    const section = container.closest('.diary-section');
    if (!section) return;

    const title = section.querySelector('.section-title');
    if (!title) return;

    const btn = document.createElement('button');
    btn.textContent = '+ 生成卡片';
    btn.style.cssText = `
        float: right;
        padding: 4px 12px;
        background: var(--primary-light);
        color: var(--primary-dark);
        border: none;
        border-radius: 6px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
    `;
    btn.onmouseover = () => { btn.style.background = 'var(--primary)'; btn.style.color = 'white'; };
    btn.onmouseout = () => { btn.style.background = 'var(--primary-light)'; btn.style.color = 'var(--primary-dark)'; };
    btn.onclick = generateNewCard;
    title.appendChild(btn);
}

async function generateNewCard() {
    const container = document.getElementById('now-cards');
    if (!container) return;

    const result = await FlowerAPI.post('/花朵日记/api/generate-card');
    if (result && result.success) {
        // 移除占位卡片（灰色提示）
        const placeholder = container.querySelector('[style*="border-left-color: var(--text-muted)"]');
        if (placeholder) placeholder.remove();

        // 创建新卡片 DOM
        const card = document.createElement('div');
        card.className = 'now-card';
        card.innerHTML = `
            <div class="card-time">${result.card.time}</div>
            <div class="card-text">${result.card.text}</div>
        `;
        container.appendChild(card);

        showToast('🌱 新的状态卡片已生成');
    }
}

// ============ Utility Functions ============
// 格式化时间
function formatTime(date) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

// ============ Pull to Go Home (Mobile Sub-pages) ============
(function initPullToGoHome() {
    // 只在移动端 + 非首页生效
    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    const isHomePage = window.location.pathname === '/' ||
                       window.location.pathname === '' ||
                       document.body.classList.contains('home-page');

    if (!isMobile || isHomePage) return;

    let startY = 0;
    let currentY = 0;
    let isPulling = false;
    const THRESHOLD = 80; // 触发阈值
    const mainContent = document.querySelector('.main-content') || document.body;

    // 只在页面滚动到顶部时才开始
    document.addEventListener('touchstart', (e) => {
        if (window.scrollY <= 0) {
            startY = e.touches[0].clientY;
            isPulling = true;
        }
    }, { passive: true });

    document.addEventListener('touchmove', (e) => {
        if (!isPulling) return;

        currentY = e.touches[0].clientY;
        const deltaY = currentY - startY;

        if (deltaY > 0 && window.scrollY <= 0) {
            // 页面跟随手指下移（带阻尼效果）
            const dampedDelta = deltaY * 0.4; // 阻尼系数
            mainContent.style.transition = 'none';
            mainContent.style.transform = `translateY(${dampedDelta}px)`;

            // 更新提示
            const hint = document.getElementById('pull-home-hint');
            if (hint) {
                // 跟手移动时禁用过渡动画
                hint.style.transition = 'none';
                // 提示元素从顶部滑入视图
                const hintProgress = Math.min(deltaY / THRESHOLD, 1);
                hint.style.opacity = hintProgress;
                hint.style.transform = `translateY(${-100 + hintProgress * 100}%)`;
                if (deltaY > THRESHOLD) {
                    hint.classList.add('active');
                    hint.querySelector('span').textContent = '释放返回首页';
                } else {
                    hint.classList.remove('active');
                    hint.querySelector('span').textContent = '下拉返回首页';
                }
            }
        }
    }, { passive: true });

    document.addEventListener('touchend', () => {
        if (!isPulling) return;

        const deltaY = currentY - startY;

        if (deltaY > THRESHOLD) {
            // 触发返回首页动画
            triggerGoHomeAnimation();
        } else {
            // 弹性回弹
            mainContent.style.transition = 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            mainContent.style.transform = 'translateY(0)';
            // 隐藏提示（滑回顶部之外）
            const hint = document.getElementById('pull-home-hint');
            if (hint) {
                hint.classList.remove('active');
                // 恢复过渡动画，平滑滑回顶部
                hint.style.transition = 'opacity 0.2s ease, transform 0.25s ease';
                hint.style.opacity = '0';
                hint.style.transform = 'translateY(-100%)';
            }
        }

        isPulling = false;
        startY = 0;
        currentY = 0;
    });

    function triggerGoHomeAnimation() {
        // 页面整体向下滑出
        mainContent.style.transition = 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.4s ease';
        mainContent.style.transform = 'translateY(100vh)';
        mainContent.style.opacity = '0';

        // 隐藏提示（滑回顶部之外）
        const hint = document.getElementById('pull-home-hint');
        if (hint) {
            hint.style.transition = 'opacity 0.2s ease, transform 0.25s ease';
            hint.style.opacity = '0';
            hint.style.transform = 'translateY(-100%)';
        }

        // 动画结束后跳转
        setTimeout(() => {
            window.location.href = '/';
        }, 500);
    }
})();
