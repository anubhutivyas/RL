# Documentation Quality Tools

This directory contains tools for maintaining high-quality documentation for NeMo RL.

## Tools Overview

### `.cursorrules` - Cursor Editor Rules
A comprehensive set of rules for the Cursor editor that provides real-time suggestions for:
- Spelling and grammar corrections
- Punctuation consistency
- Technical terminology accuracy
- Style guidelines
- Documentation structure

**Usage**: The `.cursorrules` file is automatically loaded by Cursor and provides real-time suggestions as you edit documentation files.

## Quality Guidelines

### Common Issues to Check

#### Spelling
- **"NeMo RL"** (correct capitalization in text, not code/links)
- **"fine-tuning"** (hyphenated)
- **"state-of-the-art"** (hyphenated)

#### Grammar
- **"This guide walks you through"** (active voice)
- **"Use this to"** (more direct)

#### Punctuation
- **"DPO, GRPO, and SFT"** (Oxford comma)
- **"For example,"** (comma instead of colon)

#### Style
- **"such as DPO, GRPO"** (more formal than "like")

## Quality Checklist

When reviewing documentation, check for:

- [ ] **Spelling accuracy** - All technical terms spelled correctly
- [ ] **Grammar correctness** - Proper sentence structure and verb agreement
- [ ] **Punctuation consistency** - Oxford commas, proper quotation marks
- [ ] **Technical accuracy** - Code examples work, links are valid
- [ ] **Clear and concise writing** - No run-on sentences or unclear phrasing
- [ ] **Proper formatting** - Consistent markdown formatting
- [ ] **Working links** - All internal and external links are valid
- [ ] **Consistent terminology** - Same terms used throughout
- [ ] **Appropriate difficulty level** - Content matches target audience
- [ ] **Complete frontmatter** - All required metadata present

## Technical Terms

The `.cursorrules` validates these technical terms:
- **NeMo RL** (not "nemo rl" or "nemo-rl" in text)
- **DPO** (Direct Preference Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **SFT** (Supervised Fine-Tuning)
- **Hugging Face** (not "huggingface")
- **Megatron-LM** (with hyphen)
- **Weights & Biases** (with ampersand)
- **vLLM** (correct capitalization)

## Workflow

### For Writers
1. **Write content** following the `.cursorrules` guidelines
2. **Review manually** for spelling, grammar, and style
3. **Use Cursor's suggestions** for real-time feedback
4. **Follow the quality checklist** before committing

### For Reviewers
1. **Review content** against quality guidelines
2. **Check for consistency** in terminology and style
3. **Request fixes** for spelling, grammar, or style issues
4. **Approve** only after quality standards are met

## Getting Help

- Review the `.cursorrules` file for style guidelines
- Consult the main documentation for writing standards
- Use Cursor's built-in suggestions for real-time feedback

## License

These tools are part of the NeMo RL project and follow the same license terms. 