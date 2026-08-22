/* atoms.dev/zh clone — behaviors
   Models: reveal=IntersectionObserver fade; marquee/carousel=CSS loop;
   pricing & faq & v1 accordion = click-driven; v2 nav = scroll-driven. */
(function () {
  'use strict';

  /* ---------- scroll reveal (.transitionnode.fade equivalent) ---------- */
  var revealEls = document.querySelectorAll('.reveal');
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  revealEls.forEach(function (el) { io.observe(el); });

  /* ---------- hero placeholder rotation ---------- */
  var PH = [
    '上线一个支持 Stripe 支付的电子商务网站...',
    '构建一个 SaaS 落地页并收集等待名单',
    '创建一个管理客户数据的内部工具',
    '上线一个个人作品集网站'
  ];
  var phEl = document.getElementById('heroPh');
  var phI = 0;
  if (phEl) setInterval(function () {
    phI = (phI + 1) % PH.length;
    phEl.style.opacity = '0';
    setTimeout(function () { phEl.textContent = PH[phI]; phEl.style.opacity = '1'; }, 300);
  }, 3200);

  /* ---------- logo wall: duplicate for seamless 28s marquee ---------- */
  var logoTrack = document.getElementById('logoTrack');
  if (logoTrack) logoTrack.innerHTML += logoTrack.innerHTML;

  /* ---------- AI team carousel: build 8 cards x2 ---------- */
  var AGENTS = [
    { role: 'Deep Researcher', name: 'Iris', color: 'rgb(181,126,220)', img: 'assets/48-iris.cWsNABAt.png', desc: '通过 Deep Research 发现真实需求和细分市场，然后将信号转化为聚焦的机会。' },
    { role: 'Architect', name: 'Bob', color: 'rgb(124,134,216)', img: 'assets/49-bob.CK05J5j-.png', desc: '设计系统蓝图，选择合适的结构，使你的应用可扩展且可靠。' },
    { role: 'Ads Specialist', name: 'Adrian', color: 'rgb(232,163,61)', img: 'assets/50-adrian.B6ZAN_wb.png', desc: '自动运行 Google Ads。Ads Agent 负责管理广告系列创建、跟踪和优化，让你以更少的投入实现增长扩展。' },
    { role: 'Product Manager', name: 'Emma', color: 'rgb(242,160,198)', img: null, desc: '将你的想法转化为明确的规格和范围，以便构建保持简单且可用。' },
    { role: 'Team Leader', name: 'Mike', color: 'rgb(85,179,245)', img: null, desc: '端到端运行计划，协调 agents，并请求你的批准，这样你在保持知情的同时也能快速行动。' },
    { role: 'SEO Specialist', name: 'Sarah', color: 'rgb(55,183,164)', img: null, desc: '快速推出 SEO 页面并自动化优化，以更低的成本快速带来自然流量。' },
    { role: 'Engineer', name: 'Alex', color: 'rgb(66,103,255)', img: null, desc: '通过连接前端、后端、集成和部署，构建一个可投入生产的全栈应用。' },
    { role: 'Data Analyst', name: 'David', color: 'rgb(52,168,83)', img: 'assets/51-david.BH0CUhJj.png', desc: '通过分析海量数据发现增长机会。并呈现清晰洞察，帮助你做出更明智、数据驱动的决策。' }
  ];
  var track = document.getElementById('agentTrack');
  if (track) {
    var html = AGENTS.map(function (a) {
      var banner = a.img
        ? '<div class="agent-banner" style="background:' + a.color + '"><img src="' + a.img + '" alt=""></div>'
        : '<div class="agent-banner" style="background:' + a.color + '"><span class="name">' + a.name + '</span></div>';
      return '<div class="agent-card">' + banner +
        '<h3>' + a.role + '</h3><p>' + a.desc + '</p>' +
        '<span class="agent-plus">+</span></div>';
    }).join('');
    track.innerHTML = html + html; /* x2 for seamless loop */
  }

  /* ---------- value1 accordion ---------- */
  var v1 = document.getElementById('v1Accordion');
  if (v1) v1.addEventListener('click', function (e) {
    var item = e.target.closest('.v1-item');
    if (!item) return;
    v1.querySelectorAll('.v1-item').forEach(function (i) { i.classList.remove('open'); });
    item.classList.add('open');
  });

  /* ---------- value2 scroll-driven nav (IntersectionObserver, not clicks) ---------- */
  var v2Nav = document.getElementById('v2Nav');
  var v2Panels = document.getElementById('v2Panels');
  if (v2Nav && v2Panels) {
    var navBtns = v2Nav.querySelectorAll('button[data-p]');
    var panels = v2Panels.querySelectorAll('.v2-panel');
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var idx = Array.prototype.indexOf.call(panels, en.target);
        if (idx < 0 || idx >= navBtns.length) return;
        navBtns.forEach(function (b) { b.classList.remove('active'); });
        navBtns[idx].classList.add('active');
      });
    }, { rootMargin: '-40% 0px -50% 0px' });
    panels.forEach(function (p, i) { if (i < navBtns.length) spy.observe(p); });
    navBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        panels[+b.dataset.p].scrollIntoView({ block: 'center' });
      });
    });
  }

  /* ---------- pricing toggle (click-driven, verified values) ---------- */
  var toggle = document.querySelector('.p-toggle');
  if (toggle) toggle.addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-mode]');
    if (!btn || btn.classList.contains('active')) return;
    toggle.querySelectorAll('button').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var yearly = btn.dataset.mode === 'yearly';
    document.querySelectorAll('.p-price .amt').forEach(function (a) {
      if (!a.dataset.monthly) return; /* Free $0 has no data attrs */
      a.textContent = yearly ? a.dataset.yearly : a.dataset.monthly;
    });
    document.querySelectorAll('[data-was]').forEach(function (w) {
      w.style.display = yearly ? '' : 'none';
    });
  });

  /* ---------- FAQ accordion ---------- */
  var faq = document.getElementById('faqList');
  if (faq) faq.addEventListener('click', function (e) {
    var q = e.target.closest('.faq-q');
    if (!q) return;
    var item = q.parentElement;
    var wasOpen = item.classList.contains('open');
    faq.querySelectorAll('.faq-item').forEach(function (i) { i.classList.remove('open'); });
    if (!wasOpen) item.classList.add('open');
  });
})();
