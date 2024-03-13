---
title: Cursor编辑器使用Chatgpt API
mathjax: false
date: 2023-12-05 11:30:55
tags:
  - [Chatgpt, Cursor IDE, AI辅助编程]
categories:
  - [深度学习, Chatgpt]
---

我最近发现 Cursor 这个编辑器很好，基于开源的 vscode，速度非常快，插件也很给力。在 MacOS 上用起来比较丝滑。

更重要的是，有了 Cursor 的账户，就可以有总共40条 GPT4 和200条/月 GPT3.5 的AI辅助编程可以使用。无论用来 coding 还是用 markdown 写小作文，都挺好用的。——省去了切换浏览器的工作，Cursor 也会把你编辑器的上下文直接 hint 给 Chatgpt。

Cursor 的 Pro 版本，每个月$20，年付有打折。这对于 coder 来说是一个不错的选择，因为 OpenAI 支持 GPT4 的 plus 账户还要排队不是。

此外，我试了试在 Cursor 的免费版本中登记自己的 API Key，自己向 OpenAI 付费的方式。原想这种方式大概是省钱的，by token pay as you go 对不对。但是实际用下来比较氪金。因为 Cursor 总会把你当前编辑的代码作为 context token 给到 API。我看了一下 OpenAI 的 billing 记录，用量挺吓人的，特别是在一页 code 比较长的时候。你也可以选择 Return without context 省点 token 费用，但这样使用体验太差了。

结论是不要用自己的 API Key。