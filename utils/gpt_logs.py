import logging


class GPTLogs:
    def __init__(self,
                 file_name='train.log',
                 log_name='train_logger',
                 log_format='%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s'):
        self.file_name = file_name
        self.log_name = log_name
        self.log_format = log_format
        self.logger = None
        self.logs = None

    def create_logs(self):
        # 创建logger对象
        self.logger = logging.getLogger(self.log_name)

        # 设置日志等级
        self.logger.setLevel(logging.DEBUG)

        # 追加写入文件a ，设置utf-8编码防止中文写入乱码
        self.logs = logging.FileHandler(self.file_name, 'a', encoding='utf-8')

        # 向文件输出的日志级别
        self.logs.setLevel(logging.DEBUG)

        # 向文件输出的日志信息格式
        formatter = logging.Formatter(self.log_format)

        self.logs.setFormatter(formatter)

        # 加载文件到logger对象中
        self.logger.addHandler(self.logs)

    def remove_handler(self):
        self.logger.removeHandler(self.logs)
