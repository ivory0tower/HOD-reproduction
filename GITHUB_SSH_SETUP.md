# GitHub SSH 配置和上传指南

## 当前状态
项目已准备好上传到GitHub，但需要配置SSH密钥认证。

## SSH密钥已生成
SSH密钥对已在服务器上生成：
- 私钥：`~/.ssh/id_rsa`
- 公钥：`~/.ssh/id_rsa.pub`

## 需要完成的步骤

### 1. 添加SSH公钥到GitHub账户

**公钥内容（需要复制到GitHub）：**
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDRLeSIx7xOgrR3m1Y8JIl94j/SNa14B62clhdF8hOcSCZnlAnxhoFrfDlZ8tFFJ/Xx3xQZ0w772Drw0pA39kbgZiDMlaVWp66yL0bwH3XFAIQ2fT/0OEOLV41UyQgnTlaE07hmdvj2t4+m3+kqKF3+LkFpD9b37eAeSH65UEqASD99MqXBsDOuShpDGFvnx6ZbCZbyy84iBPIwKjUpCv6r3m4irNYEePzbFKSKA5Cur7O7ny97CyGOAzVgcel9ryMFccejo6JbI4ORuCFGcBBsgkxgRVdBQSr970tUWK2bRx2nZyh+bzCQaBiQc8IDZHjWXizW6dZrzn/9jJSs9w1ISYG3Oj2pQlh3nsy7yrj0HXhbXS3t4eT68ta81lu/B1JoO5ZpWnUWG99CitEJNK5ZDss+NfdA/n3GZMT29Ug5tLZOP2C5G58reFpydEhh/Z0tSaRk1UHjICs5bd53KPBOO636MwC5Eb6w+UQpCEOSdnNYIrnfzv88XafcgkI0pgoA4MadQ4ZPedpe/EXhNtW+QvUEpHibm2+O60R+cK/UwiTng9iOqs3sNMkqrhL9k43Mn/RIxqNE7ZtWxHgV52zFhA1JVy2PXHNI7YYlRBvkq+VrybOVk2XbGWSVwXM+zmtpIy+ZXKp2Q299R5lRmOBGng47uwlQCXlYfjE1xiuW5w== hod@example.com
```

**操作步骤：**
1. 登录GitHub账户
2. 点击右上角头像 → Settings
3. 左侧菜单选择 "SSH and GPG keys"
4. 点击 "New SSH key"
5. Title填写：`HOD-reproduction-server`
6. Key类型选择：`Authentication Key`
7. 将上面的公钥内容完整复制到Key字段
8. 点击 "Add SSH key"

### 2. 测试SSH连接
添加SSH密钥后，在终端运行以下命令测试连接：
```bash
ssh -T git@github.com
```

### 3. 推送代码到GitHub
连接测试成功后，运行：
```bash
cd /root/autodl-tmp/HOD_reproduction
git push -u origin main
```

## 故障排除

### 如果SSH连接失败
1. 确认SSH密钥已正确添加到GitHub
2. 检查SSH代理：
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   ```

### 如果推送失败
1. 确认仓库存在且有写入权限
2. 检查分支名称是否正确
3. 尝试强制推送（谨慎使用）：
   ```bash
   git push -f origin main
   ```

## 项目信息
- **仓库地址**: https://github.com/ivory0tower/HOD-reproduction
- **本地路径**: `/root/autodl-tmp/HOD_reproduction`
- **远程配置**: SSH格式 (`git@github.com:ivory0tower/HOD-reproduction.git`)
- **当前分支**: main

## 完成后验证
上传成功后，可以在GitHub仓库页面看到所有文件，包括：
- README.md（包含数据集信息）
- 训练脚本和模型文件
- 数据处理工具
- 项目文档

注意：由于.gitignore配置，大型数据集文件不会被上传，这是正常的。