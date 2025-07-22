import wandb

# 登录到 wandb
wandb.login()

# 设置项目和运行信息
project_name = "verl_grpo_cosreward"  # 替换为您的项目名称
run_id = "5xg3hko"  # 替换为运行的 ID

# 加载运行
api = wandb.Api()
run = api.run(f"{project_name}/{run_id}")

# 列出文件
for file in run.files():
    print(file.name)  # 列出运行中所有文件名以确认

# 下载日志文件
log_file = run.file("output.log")  # 替换为实际文件名
log_file.download(replace=True)  # 下载到当前目录
