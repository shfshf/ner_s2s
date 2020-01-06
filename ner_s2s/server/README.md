## HTTP server
### run service
```bash
python -m ner_s2s.server.http /path/to/saved_model
```

默认启动在 主机： `localhost` 端口：`5001`
### input format
example:
```
http://localhost:5001/parse?q=播放周杰伦的叶惠美
```
in HTTP format
```text
GET /parse?q=播放周杰伦的叶惠美 HTTP/1.1
Host: howl-MS-7A67:5001
```
### output
example:
```json
{
    "ents": [
        "人名",
        "歌曲名"
    ],
    "spans": [
        {
            "end": 5,
            "start": 2,
            "type": "人名"
        },
        {
            "end": 9,
            "start": 6,
            "type": "歌曲名"
        }
    ],
    "text": "播放周杰伦的叶惠美"
}
```
