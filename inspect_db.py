import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('data/memory_system.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute('SELECT * FROM memories ORDER BY id')
mems = [dict(r) for r in cur.fetchall()]

print('--- ALL MEMORIES ---')
for m in mems:
    print(f"ID {m['id']}: Subject={m['subject']} | Pred={m['predicate']} | Obj={m['object_val']} | Cat={m['category']} | Status={m['status']} | Turn={m['source_turn_id']} | Created={m['created_at']}")
    print(f"   Content: {m['content']}")

cur.execute('SELECT * FROM conversation_turns WHERE id >= 20 ORDER BY id')
turns = [dict(r) for r in cur.fetchall()]
print('\n--- CONVERSATION TURNS >= 20 ---')
for t in turns:
    resp = (t['assistant_response'] or '')[:80].replace('\n', ' ')
    print(f"Turn {t['id']}: User='{t['user_message']}' | Assistant='{resp}...'")

cur.execute('SELECT * FROM memory_conflict_logs WHERE old_memory_id BETWEEN 29 AND 35 OR new_memory_id BETWEEN 29 AND 35')
logs = [dict(r) for r in cur.fetchall()]
print('\n--- CONFLICT LOGS ---')
for l in logs:
    print(dict(l))
