kubectl apply -f pv.yaml -n 1253431691-emr-jktvb2qj

kubectl apply -f pvc.yaml -n 1253431691-emr-jktvb2qj

kubectl apply -f test.yaml -n 1253431691-emr-jktvb2qj

kc get pod -n 1253431691-emr-jktvb2qj

kc delete pod nfs-test-pod -n 1253431691-emr-jktvb2qj

kc describe pod nfs-test-pod -n 1253431691-emr-jktvb2qj

kubectl get pv demo-nfs-pv -n 1253431691-emr-jktvb2qj
kubectl get pvc demo-nfs-pvc -n 1253431691-emr-jktvb2qj

kubectl delete pvc demo-nfs-pvc  -n 1253431691-emr-jktvb2qj