FROM ds-runtime:1.0

COPY ./dolphinscheduler.keytab /data/service/dolphinscheduler.keytab
COPY ./krb5.conf /etc/krb5.conf
COPY ./spark-docker-python.py /data/src/spark-docker-python.py
COPY ./hadoop_conf /etc/hadoop/conf

WORKDIR /app

# 执行一次 kinit 命令
RUN kinit -kt /data/service/dolphinscheduler.keytab dolphinscheduler@INSIGHT.TQCBJ.NIO.COM || true

# 启动时执行你要的主程序（这里你可以换成具体的 ENTRYPOINT 或脚本）
CMD ["tail", "-f", "/dev/null"]