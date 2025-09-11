#include <bits/stdc++.h>
using namespace std;
const long long Mod = 1e9 + 7, N = 200005;
long long tree[N << 1];
void add(int idx, long long val) {
  while (idx <= 2 * N) {
    tree[idx] += val;
    idx += idx & (-idx);
  }
}
long long get(int idx) {
  long long ret = 0;
  while (idx > 0) {
    ret += tree[idx];
    idx -= idx & (-idx);
  }
  return ret;
}
int up[22][N], depth[N];
vector<int> v[N];
int sz = 1;
int l[N], f[N];
long long c[N];
void dfs(int u, int p) {
  f[u] = sz;
  sz++;
  up[0][u] = p;
  for (auto r : v[u]) {
    if (r != p) {
      depth[r] = depth[u] + 1;
      dfs(r, u);
    }
  }
  l[u] = sz;
  sz++;
}
int get_LCA(int a, int b) {
  if (depth[a] < depth[b]) swap(a, b);
  int k = depth[a] - depth[b];
  for (int j = 18; j >= 0; j--) {
    if ((k >> j) & 1) a = up[j][a];
  }
  if (a == b) return a;
  for (int j = 18; j >= 0; j--) {
    if (up[j][a] != up[j][b]) {
      a = up[j][a];
      b = up[j][b];
    }
  }
  return up[0][a];
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int t, q, x, n, y;
  t = 1;
  for (int w = 1; w <= t; w++) {
    cin >> n >> q;
    for (int i = 1; i <= n; i++) {
      cin >> c[i];
      c[i] = abs(c[i]);
    }
    for (int i = 0; i < n - 1; i++) {
      cin >> x >> y;
      v[x].push_back(y);
      v[y].push_back(x);
    }
    sz = 1;
    depth[1] = 1;
    dfs(1, 1);
    for (int j = 1; j <= 18; j++) {
      for (int i = 1; i <= n; i++) {
        up[j][i] = up[j - 1][up[j - 1][i]];
      }
    }
    cout << '\n';
    for (int i = 1; i <= n; i++) {
      add(f[i], c[i]);
      add(l[i], -c[i]);
    }
    while (q--) {
      cin >> x;
      if (x == 2) {
        cin >> x >> y;
        int z = get_LCA(x, y);
        long long ans = get(f[x]) + get(f[y]) - 2 * get(f[z]) + c[z];
        if (z != x && z != y) {
          x = up[0][x];
          y = up[0][y];
          int zz = get_LCA(x, y);
          ans += get(f[x]) + get(f[y]) - 2 * get(f[z]) + c[z];
          cout << ans << '\n';
        } else if (x == z) {
          ans += get(f[up[0][y]]) - get(f[x]);
          cout << ans << '\n';
        } else if (y == z) {
          ans += get(f[up[0][x]]) - get(f[y]);
          cout << ans << '\n';
        }
      } else {
        cin >> x >> y;
        y = abs(y);
        int pre = c[x];
        c[x] = y;
        add(f[x], y - pre);
        add(l[x], pre - y);
      }
    }
  }
  return 0;
}
