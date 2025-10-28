<div align="center">
<img src="assets/figs/icon.png" width="300">
<h1>🐈 🍵 Meow-Tea-Taro 💜: A Modular Multi-turn Agentic RL Framework with Configurable 🌎Environments, 🤖Policies, and ⭐Rewards</h1></div>

<p align="center">
  <a href="https://arxiv.org/abs/2510.01132">
    <img src="https://img.shields.io/badge/Paper-arXiv%202510.11701-b31b1b?logo=arxiv&logoColor=white" alt="Paper on arXiv" />
  </a>
<a href="https://meow-tea-taro.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Docs-Documentation-blue?logo=readthedocs&logoColor=white" alt="Documentation" />
  </a>
  <a href="https://huggingface.co/datasets/PEARLS-Lab/meow-tea-taro-dataset">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green?logo=apache&logoColor=white" alt="Apache License" />
  </a>
  <a href="https://github.com/pearls-lab/meow-tea-taro">
    <img src="https://img.shields.io/github/stars/pearls-lab/meow-tea-taro?style=social" alt="GitHub stars" />
</a>
</p>

<div align="center">
<p style="font-size: 18px;">
<strong>The codebase for paper: <a href="https://arxiv.org/abs/2510.01132">A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning</a></strong>
</p>
</div>

## Table of Contents
📖 [Overview](#overview)  
🚀 [Quick Start](#quick-start)  
🔔 [News](#news)  
📈 [Recipes](#recipes)  
🔑 [Key Findings](#key-findings)  
🧋 [Build Your Own Agentic Pipelines!](#build-your-own-agentic-pipelines)  
📚 [Citation](#citation)  

## 📖 Overview
**Welcome to the Meow-Tea Café! ☕️**

Just as "Multi" sounds like "Meow-tea" 🐈 🍵, our RL framework brings together the best ingredients for cooking up powerful Multi-turn Agentic RL solutions. 

### What's on the Menu?
At Meow-Tea Café, we serve a diverse menu of **agentic dishes** (tasks) ranging from text-based adventures to real-world software engineering challenges. Each dish in our [`meow_tea_train/agentic_menu/`](./meow_tea_train/agentic_menu/) represents a different agentic task:

- 🎮 **TextWorld**: Text-based adventure game environments
- 🏠 **ALFWorld**: Situated household tasks
- 💻 **SWE-Gym**: Realworld software engineering problems
- *...and more specialty dishes coming soon!*


### The Art of RL Cooking 👨‍🍳
For each dish on our menu, we identify three essential **RL cooking processes** that can bring out its best flavors. We've made these components **fully configurable** in our framework:

1. **🌎 Environments** - The foundation of your dish (the agentic task itself, sync vs async rollout, tool use, thinking abilities)
2. **🤖 Policies** - Your RL cooking technique (PPO, GRPO, RLOO, and more)
3. **⭐ Rewards** - The perfect heat control and timing (single vs dense rewards, verified vs learned rewards)

These three pillars of RL cooking are the heart of our framework, allowing **RL cooking lovers** - whether you're a researcher, practitioner, or student - to experiment and explore innovative ways to think about and solve agentic tasks and challenges.

Can't find your favorite dish on our menu? No problem! The Meow-Tea Café includes a special [**`build_your_own`**](./meow_tea_train/agentic_menu/build_your_own/) section, where we'll walk you through creating the agentic task (dish) you want to cook. 


### Our Recipes and Research Insights
We provide **tested recipes for different** agentic tasks. These recipes are training configurations that have been validated through our experiments. You can find them under [`recipes/`](./recipes/) and [`examples/{agentic_task}`](./examples/). 

Our paper presents systematic findings on **what works and what doesn't** for multi-turn agentic RL. Key research questions we address include:

- *Can we train agents on simpler environments and expect them to perform well on complex ones?*
- *How do different RL algorithms impact multi-turn RL training?*
- *Is there an optimal ratio of SFT:RL data given fixed budget?*
- *How does the density of rewards impact multi-turn RL training?*
- *...and more in our paper*


## 🚀 Quick Start

Start by creating and running a Docker container with GPU support
```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace --name meow-tea-taro hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 sleep infinity
docker start meow-tea-taro
docker exec -it meow-tea-taro bash
```

Clone the latest meow-tea-taro repository and install:
```bash
git clone git@github.com:pearls-lab/meow-tea-taro.git
cd meow-tea-taro
pip install -e .
```
That's it!

Run a quick example of multi-turn PPO on TextWorld tasks using Qwen2.5-0.5B-Instruct: 
```bash
sh examples/textworld/run_ppo_qwen-0.5b.sh
```
You should be able to see the training curve like this: [wandb log](https://api.wandb.ai/links/pearls-lab/wurzotla).


Now you are ready to cook your RL dishes! Refer to [**the meow-tea-taro documentation**](https://meow-tea-taro.readthedocs.io/en/latest/index.html) for detailed environment, policy and reward configuration tutorials.

The datasets used in our [`meow_tea_experiments`](./meow_tea_experiments/) are available in 🤗 Huggingface: [PEARLS-Lab/meow-tea-taro-dataset](https://huggingface.co/datasets/PEARLS-Lab/meow-tea-taro-dataset).


## 🔔 News
- 🎉 **[10/21/2025]** Meow-Tea-Taro codebase is now open-source! Recipes, datasets, and model checkpoints are available.
- 🎉 **[10/01/2025]** Paper "A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning" released

## 📈 Recipes

We share the recipes for TextWorld, ALFWorld, and SWE-Gym tasks [here](./recipes/). The table records a summary of the agentic tasks, our configuration, and the performance of our recipes.

| Agentic Task | Base Model | Policy | Reward | Success Rate | Performance Delta| Script |
|-------------|-------------|---------|---------|--------------|---------------|---------------|
| TextWorld (w2-o3-q4) | Qwen2.5-1.5B-Instruct | PPO | Single | 97% | ⬆️ 82% | [recipe](./recipes/textworld_w2-o3-q4_ppo.sh) |
| TextWorld (w4-o6-q8) | Qwen2.5-1.5B-Instruct | PPO | Single | 94% | ⬆️ 93% | [recipe](./recipes/textworld_w4-o6-q8_ppo.sh) |
| TextWorld (cooking) | Qwen2.5-7B-Instruct | PPO | Dense | 58% | ⬆️ 29% | [recipe](./recipes/textworld_cooking_ppo.sh)
| TextWorld (cooking) | Qwen2.5-7B-Instruct | RLOO | Dense | 55% | ⬆️ 26% | [recipe](./recipes/textworld_cooking_rloo.sh)
| Alfworld (text-based) | Qwen2.5-7B-Instruct | PPO | Single | 74% | ⬆️ 73% | [recipe (sft)](./recipes/alfworld_text_sft.sh), [recipe (ppo)](./recipes/alfworld_text_ppo.sh)
| SWE-Gym | Qwen3-8B | GRPO | Single | 22% | ⬆️ 18% | [recipe](./recipes/swegym_grpo.sh)


## 🔑 Key Findings
Check out paper for key finds and takeaways: [**A Pracititioner's Guide to Multi-turn Agentic Reinforcement Learning**](https://arxiv.org/abs/2510.01132).

We are committed to expanding the codebase (adding more agentic tasks, more RL algorithms, and different reward modeling techniques). We will provide research insights in our subsequent experiments on agentic multi-turn RL along the way. Stay tuned!

## 🧋 Build Your Own Agentic Pipelines!
Tutorials are under development. Will release real soon! 

## 📚 Citation
```bibtex
@misc{wang2025practitionersguidemultiturnagentic,
      title={A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning}, 
      author={Ruiyi Wang and Prithviraj Ammanabrolu},
      year={2025},
      eprint={2510.01132},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.01132}, 
}
```

---

**Join our community of RL chefs! 👨‍🍳👩‍🍳** At Meow-Tea Café, we're passionate about **promoting open-source RL recipes and models**. We welcome contributions, new recipes, and fresh ideas to make Meow-Tea Café even better. 

**Stay tuned as we keep expanding our menu and refining our recipes!**
