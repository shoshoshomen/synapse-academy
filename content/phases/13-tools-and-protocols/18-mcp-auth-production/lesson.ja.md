# 本番環境のMCP認証 — エンロールメント、JWKSリフレッシュ、オーディエンスピン留めトークン

> Lesson 16はOAuth 2.1ステートマシンをメモリ上で構築した。2026年には、実際の組織に出荷するすべてのMCPサーバーが本番認証の下に位置する：クライアントID メタデータドキュメント（CIMD）を優先し下位互換のフォールバックとして動的クライアント登録を使う無制限のクライアント母集団に対応したクライアントエンロールメント、認可サーバーメタデータ発見（RFC 8414またはOpenID Connect Discovery）、午前3時のトークン検証を壊さないJWKSキャッシュリフレッシュ、クロスリソースのリプレイを拒否するオーディエンスピン留めトークンだ。このレッスンでは認可サーバー、リソースサーバー（MCPサーバー）、クライアントの3つのロールで完全なサーフェスをモデリングし、発見から検証済みツール呼び出しまでのすべてのホップをトレースできるようにする。
>
> **仕様注記（2025年11月25日）：** 2025年11月のMCP認可仕様は動的クライアント登録を `SHOULD` から `MAY` に降格させ、**クライアントIDメタデータドキュメント（CIMD）**を推奨デフォルトのエンロールメントメカニズムとした。このレッスンでは仕様の優先順位に従って両方を教えるが、コードは1つのプロセスで完全に自己完結しているためDCRのウォークスルーを保持している。


## 学習目標

- RFC 8414メタデータを通じて認可サーバーを発見し、コントラクトを検証できる。
- MCPクライアントが管理者の介入なしにエンロールできるようにRFC 7591動的クライアント登録を実装できる。
- キー更新後もシグネチャ検証が機能するようにJWKSキーをスケジュールに従ってキャッシュおよびリフレッシュできる。
- RFC 8707リソースインジケーターを使用してトークンを単一のMCPリソースにピン留めし、Confused-deputyの再利用を拒否できる。
- 3つのロールを明確に分離できる — 認可サーバー、リソースサーバー、クライアント — それぞれが自分に属するチェックだけを実施する。
- IdP機能マトリクスを読み、IdPがMCPの認証プロファイルを満たせない場合にデプロイを拒否できる。

## 問題背景

Lesson 16のシミュレーターはOAuth 2.1をメモリ上で動作させる。本番環境にはメモリのみのシミュレーターでは見えない3つの運用上のギャップがある。

最初のギャップはエンロールメントだ。実際の組織では数百のMCPサーバーと数千のMCPクライアントが動作している。オペレーターはすべてのCursorユーザーをOAuthクライアントとして手動で登録しない。2025年11月25日の仕様はクライアントに優先順序を与えている：事前登録済みの `client_id` があればそれを使う、なければ**クライアントIDメタデータドキュメント**（クライアントが管理するHTTPS URLを `client_id` として使用し、認可サーバーがオンデマンドでメタデータを*プル*する）を使う、なければ**RFC 7591動的クライアント登録**（クライアントが `POST /register` を*プッシュ*してその場で `client_id` を受け取る）にフォールバックする、それ以外はユーザーにプロンプトを出す。CIMDはDNSをルートにした信頼モデルを維持しながらサーバーごとの登録を完全になくすため推奨デフォルトだ。DCRは下位互換性のために残されている。どちらも認可サーバーのメタデータからエントリポイントを発見する：CIMDは `client_id_metadata_document_supported`、DCRは `registration_endpoint`。

2番目のギャップはキーローテーションだ。JWT検証は認可サーバーの署名キーに依存し、JSONウェブキーセット（JWKS）として公開される。認可サーバーはスケジュールに従ってこれらをローテーションする（多くの場合1時間ごと、インシデント対応時はより速く）。起動時に一度だけJWKSを取得するMCPサーバーはローテーションウィンドウまで正常に検証するが、その後再起動するまですべてのリクエストが失敗する。本番環境ではJWKSをキャッシュされた値として設定し、前のキーが期限切れになる前にキャッシュを上書きするリフレッシュジョブを設定し、キャッシュより新しいキーで署名されたトークンが届いた場合のキャッシュミスのフォールバックフェッチを行う。

3番目のギャップはオーディエンスバインディングだ。Lesson 16はRFC 8707リソースインジケーターを導入した。本番環境では、そのインジケーターがすべてのリクエストでのハードなクレームチェックになる。MCPサーバーはすべてのリクエストを処理する前に `token.aud` を自身の正規リソースURLと比較し、不一致を401で拒否する。これが同じ信頼メッシュ内の別のサーバーに対してトークンをリプレイするアップストリームMCPサーバー（または1つのサーバー向けのトークンを保持する悪意のあるクライアント）に対するプロトコル層での唯一の防御だ。

## コンセプト

### RFC 8414 — OAuth認可サーバーメタデータ

`/.well-known/oauth-authorization-server` のドキュメントがクライアントに必要なすべてを記述する：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCPリソースURLを与えられたクライアントはディスカバリーをチェーンする：RFC 9728の `oauth-protected-resource`（リソースサーバーのドキュメント）がissuerを指定し、`oauth-authorization-server`（このRFC）がすべてのエンドポイントを指定する。クライアントは認可URLをハードコードしない。

MCPのためにIdPを信頼する前に確認するコントラクト：

- `code_challenge_methods_supported` に `S256` が含まれる（RFC 7636によるPKCE）。仕様は明確：このフィールドが**欠けている**場合、認可サーバーはPKCEをサポートしておらず、クライアントは**処理を拒否しなければならない**。
- `grant_types_supported` に `authorization_code` が含まれ、`password` と `implicit` を拒否する。
- 少なくとも1つのエンロールメントパスが告知されている：`client_id_metadata_document_supported: true`（CIMD、推奨）**または** `registration_endpoint`（RFC 7591 DCR、フォールバック）。どちらかがあればコントラクトを満たす。
- `response_types_supported` はOAuth 2.1のために正確に `["code"]` だ。

`S256` が欠けている場合、MCPサーバーはこのIdPに対するデプロイを拒否する — PKCEに劣化モードはない。どちらのエンロールメントパスも告知されておらず、事前登録済みの `client_id` もない場合も、エンロールできない。その場合はデプロイメントマニフェストが間違っており、コードではない。

### RFC 9728（再説）— 保護リソースメタデータ

Lesson 16でRFC 9728を説明した。本番環境での違い：このドキュメントがクライアントが*このMCPサーバー*から信頼される認可サーバーを見つける唯一の場所だ。単一のMCPサーバーが複数のIdP（スタッフ用と パートナー用）からのトークンを受け入れることができる。RFC 9728がそのセットを宣言し、RFC 8414が各IdPがサポートするものを文書化する。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### クライアントIDメタデータドキュメント（推奨デフォルト）

CIMDは登録を*プッシュ*から*プル*に逆転させる。認可サーバーに `client_id` を発行させる代わりに、クライアントは管理するHTTPS URLを**その** `client_id` として使用する。URLはJSONメタデータドキュメントに解決され、認可サーバーがOAuthフロー中にオンデマンドでそれを取得する。信頼はDNSをルートにしている：サーバーオペレーターが `app.example.com` を信頼する場合、`https://app.example.com/client.json` から提供されるクライアントを信頼する。登録のラウンドトリップなし、枯渇する `client_id` ネームスペースなし、同期を保持する必要のあるサーバーごとの状態なし。

クライアントがホストするメタデータドキュメント：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

ドキュメントの `client_id` 値はそれが提供されるURLと**等しくなければならない**（認可サーバーがこれを検証し、不一致は拒否される）。認可サーバーはRFC 8414メタデータで `client_id_metadata_document_supported: true` を告知することでサポートを告知する。

仕様が明確に指摘する2つのセキュリティ上の事実：

- **SSRF。** 認可サーバーが攻撃者が提供したURLを取得する。内部/管理エンドポイントへのフェッチを防ぐためにサーバーサイドリクエストフォージェリに対して防御しなければならない。
- **localhostなりすまし。** CIMDだけでは、正当なクライアントのメタデータURLを主張して任意の `localhost` リダイレクトをバインドしようとするローカル攻撃者を止めることはできない。認可サーバーは同意時にリダイレクトURIのホスト名を明確に表示し、`localhost` のみのリダイレクトには警告を出すべきだ。

CIMDはサーバーサイドの状態を必要としないため、DCRが必要とするような登録機関を構築する必要がない。クライアント側は読み取り専用だ：静的なHTTPSエンドポイントからメタデータドキュメントを提供し、認可サーバーがそれをプルする。

### RFC 7591 — 動的クライアント登録（フォールバック / 下位互換性）

DCRはフォールバックとして `MAY` になり、2025年11月25日以前のデプロイメントとCIMDをまだサポートしていないIdPとの下位互換性のために残されている。CIMDも事前登録もなしでは、すべてのMCPクライアント（Cursor、Claude Desktop、カスタムエージェント）がIdP管理者とのアウトオブバンド交換を必要とする。DCRでは、クライアントが以下を投稿する：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

サーバーは `client_id` と後の更新用の `registration_access_token` で応答する：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` はユーザーのデバイスで動作するMCPクライアントの正しいデフォルトだ。`client_id` だけを取得し、漏洩する `client_secret` はない。PKCEがパブリッククライアントに必要な所有証明を提供する。

3つの本番環境の落とし穴：

- 登録エンドポイントはソースIPでレート制限しなければならない。これなしに、敵対者が何百万もの偽登録をスクリプト化して `client_id` ネームスペースを枯渇させることができる。登録機関がリクエストを処理する前にレート制限チェックを実行する。
- `software_statement`（クライアントを保証する署名済みJWT）はいくつかのエンタープライズIdPで必要だ。レッスンのモックはこれをスキップしている；本番環境ではlocalhostリダイレクトURI以外からの署名なし登録を拒否する検証ステップを追加する。
- `registration_access_token` はハッシュとして保存しなければならず、平文ではない。このトークンの盗取は攻撃者がクライアントのリダイレクトURIを書き換えることを意味する。

### RFC 8707（再説）— リソースインジケーター

Lesson 16で形状を確立した。本番ルール：すべてのトークンリクエストに `resource=<canonical-mcp-url>` を含め、MCPサーバーはすべての呼び出しで `token.aud` を自身のリソースURLと一致するか検証する。正規URIはサーバーの*最も具体的な*識別子だ：小文字のスキームとホスト、フラグメントなし、慣例としてトレイリングスラッシュなし。パスコンポーネントはルールによって除去されない — 仕様は個々のMCPサーバーを識別するために必要な場合にそれを保持する。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443`、`https://mcp.example.com/server/mcp` はすべて有効な正規URIだ。サーバーごとに1つを選択し、`aud` を正確にそれにピン留めする。

### RFC 7636（再説）— PKCE

PKCEはOAuth 2.1で必須だ。レッスンの認可コードフローは常に `code_challenge` と `code_verifier` を含む。サーバーはベリファイアなし、またはベリファイアが保存されたチャレンジにハッシュされないトークンリクエストを拒否する。

### MCP仕様2025年11月25日認証プロファイル

MCP仕様（2025年11月25日）はMCPサーバーの認可層が何をしなければならないかについて明確だ：

- RFC 9728の保護リソースメタデータを実装し、401での `WWW-Authenticate: Bearer resource_metadata="..."` ヘッダーまたはwell-known URI `/.well-known/oauth-protected-resource`（SEP-985でヘッダーをオプションにした）でその場所を提供する。メタデータの `authorization_servers` フィールドには少なくとも1つのサーバーを**必ず**記述する。
- セッション開始時だけでなく、**すべての**リクエストで `Authorization: Bearer ...` を通じてのみトークンを受け入れる。クエリ文字列には絶対に入れない。
- リクエストごとに `aud`、`iss`、`exp`、必要なスコープを検証する。サーバーはトークンが自分のために発行されたことを検証**しなければならない**（オーディエンス）；`aud` が欠けているか不一致の場合は拒否し、ワイルドカードとして扱ってはならない。
- 401/403では `error=...`、`resource_metadata="<PRM-URL>"` パラメーター（メタデータドキュメントのURL、単なるリソースではない）、`insufficient_scope`（403）時の `scope="..."` を含む `WWW-Authenticate: Bearer` を返す。
- 認可サーバー発見はRFC 8414 OAuthメタデータまたはOpenID Connect Discovery 1.0のどちらかを受け入れる；クライアントは優先順序で両方のwell-known suffixを試みなければならない。
- クライアント（サーバーではない）が**ミックスアップ攻撃**に対して防御する：リダイレクト前に期待される `issuer` を記録し、コードを引き換える前に `iss` 認可レスポンスパラメーター（RFC 9207）を検証する。PKCEだけではミックスアップを止められない。

### IdP機能マトリクス

すべてのIdPが完全なMCPプロファイルをサポートしているわけではない。以下のマトリクスは2025年11月25日の仕様時点での実際の機能状況を文書化している。これは*デプロイゲート*であり、推奨ではない。

| IdPカテゴリ | ASメタデータ（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707リソース | RFC 7636 S256 PKCE | 注記 |
|---|---|---|---|---|---|---|
| セルフホスト（Keycloak） | あり | 対応中 | あり | あり（24.x以降） | あり | このレッスンのMCPプロファイルのリファレンスIdP；完全なDCRパス、CIMDは新仕様を追跡中 |
| エンタープライズSSO（Microsoft Entra ID） | あり | 対応中 | あり（プレミアムティア） | あり | あり | DCRの可用性はテナントティアによって異なる；デプロイ前にターゲットテナントで確認を |
| エンタープライズSSO（Okta） | あり | 対応中 | あり（Okta CIC / Auth0） | あり | あり | Auth0（現Okta CIC）でDCR利用可能；クラシックOkta組織は管理者の事前登録が必要 |
| ソーシャルログインIdP（一般） | 様々 | なし | まれ | まれ | あり | ほとんどのソーシャルIdPはクライアントを静的なパートナーとして扱う；セルフサービスのエンロールメントなし。アイデンティティソースとしてのみ使用し、上に独自のMCP対応認可サーバーを層として追加する |
| カスタム / 自作 | 依存 | 依存 | 依存 | 依存 | 依存 | 自分で出荷する場合、完全なプロファイルを出荷しCIMDを優先する。PKCEやオーディエンスバインディングをスキップするとMCP認証コントラクトが壊れる |

### JWKSリフレッシュパターン（ASでのローテーション、リソースサーバーでのリフレッシュ）

混同すると実際の本番バグになるため、2つの動詞を分けておく：

- **ローテーション**は*認可サーバー*が行うこと：新しい署名キーを作成し、JWKSに公開し、後で古いキーを廃止する。リソースサーバーはこれに関与しておらず、IdPの秘密鍵を持っていないためできない。
- **リフレッシュ**は*リソースサーバー*が行うこと：公開されたJWKSをキャッシュに再取得する。リソースサーバーが実行するJWKSアクションはこれだけだ。

本番の失敗モードは古いキャッシュだ。スケジュールされたリフレッシュジョブとキーバリューキャッシュで解決する。リソースサーバーはジョブを実行し（cron、タイマー、ランタイムが提供するもの）、固定間隔で `<issuer>/.well-known/jwks.json` を取得して `cache[issuer] = {keys, fetched_at}` を上書きする。バリデーターはそのキャッシュから読み取る。キャッシュにない `kid` を持つトークンが**1回**の同期リフレッシュをフォールバックとしてトリガーし、再確認する。これで2つのケースを同時に処理する：スケジュールされたリフレッシュと、次のスケジュールリフレッシュの前に全く新しいキーで署名されたトークンが届くキーオーバーラップウィンドウ。

フォールバックは**再フェッチであり、ローテーションではない**。キャッシュミスのパスをrotate-and-mintに接続すると2つのことが壊れる：(1)新しいキーを作成しても欠けている `kid` と一致するものが依然として生成されないためルックアップが失敗する；(2)ランダムな `kid` 値のトークンをスプレーする攻撃者が無限のキー作成を強制する — 自己引き起こしのDoS。再フェッチは冪等性があるため、偽の `kid` は最大1回の無駄なフェッチだけをコストとする。

### 検証ルーティン

MCPサーバーはツールをディスパッチする前に検証を実行する。`code/main.py` が使用する形状：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` はJWTをデコードし、JWKSキャッシュから署名キーを解決し（ミス時に一度リフレッシュ）、シグネチャを検証し、`iss` を許可リストに対して、`aud` をこのサーバーの正規リソースに対して、`exp`、必要なスコープを確認し — 最初の失敗で `WWW-Authenticate` チャレンジを返す。これをリソースサーバー上の単一のルーティンとして維持することで、すべてのエントリポイント（すべてのツール呼び出し、すべてのトランスポート）が同じチェックを通過する。

## 使ってみる

`code/main.py` はstdlib Pythonと3つのロール（`AuthorizationServer`、`ResourceServer`、`Client`）で完全な本番フローを実行する。フロー：

1. 認可サーバーが `/.well-known/oauth-authorization-server` にRFC 8414メタデータを公開する。
2. MCPクライアントがメタデータエンドポイントを呼び出し、エンロールメントオプション（CIMDの `client_id_metadata_document_supported`、DCRの `registration_endpoint`）と `S256` PKCEサポートを確認する。
3. ウォークスルーはDCRフォールバックパスを取る：クライアントが `/register`（RFC 7591）に投稿して `client_id` を受け取る。（CIMDクライアントは代わりに自身のHTTPS `client_id` URLを提示してこのステップをスキップする。）
4. MCPクライアントが `resource` インジケーター（RFC 8707）付きのPKCE保護認可コードフロー（RFC 7636）を実行する。
5. MCPクライアントが `Authorization: Bearer ...` でMCPサーバーのツールを呼び出す。
6. MCPサーバーが `validate` を実行し、JWKSキャッシュから署名キーを解決する。
7. IdPがキーをローテーションし、スケジュールされたリフレッシュがJWKSをキャッシュに再プルする。
8. 次の呼び出しが再起動なしにリフレッシュされたキーに対して検証し、前のトークンはオーバーラップウィンドウ中も検証される。
9. 別のMCPリソースに対するオーディエンスリプレイの試みが `audience mismatch` と `resource_metadata` ポインターで401を取得する。

このレッスンのJWTはHS256と共有シークレットを使用する（レッスンがstdlibのみで動作するため）。本番環境ではRS256またはEdDSAをJWKSパターンで使用する；検証ロジックは他は同一だ。

## 成果物を出す

このレッスンでは `outputs/skill-mcp-auth.md` を生成する。MCPサーバーの設定とIdPの機能セットが与えられると、このスキルはセットアップする認証サーフェスを出力する — 保護リソースメタデータ、使用するエンロールメントパス（CIMD、事前登録、またはDCRフォールバック）、JWKSリフレッシュスケジュール、スコープマッピング、IdPが完全なRFCプロファイルをサポートしない場合の拒否ルール。

## 演習

1. `code/main.py` を実行する。フローをトレースする。ステップ6でIdPがキーをローテーションし、スケジュールされた `refresh_jwks` が公開されたセットを再プルし、古いトークン（オーバーラップウィンドウ）と新しいトークンの両方が再起動なしに検証されることを確認する。

2. 保護リソースメタデータの `authorization_servers` リストに新しいIdPを追加する。新しいIdPで署名されたトークンを発行してバリデーターが受け入れることを確認する。未リストのIdPで署名されたトークンを発行してバリデーターが `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` で拒否することを確認する。

3. `register_client` にリクエストを受け入れる前に実行されるレート制限チェックを追加する。小さな辞書にIPをキーにしたトークンバケットをソースIPごとに使用する。

4. RFC 7591を読み、レッスンの `/register` ハンドラーが検証しない2つのフィールドを特定する。検証を追加する。（ヒント：`software_statement` と `redirect_uris` URIスキーム。）

5. クライアントIDメタデータドキュメントパスを追加する。`client_id` が自身のURLと等しい `client.json` を提供し、認可サーバーがそれを取得して検証する（`client_id` ≠ URLの場合は拒否）。CIMDクライアントが `register_client` 呼び出しなしにエンロールされることを確認する。

6. DoS修正を証明する。ランダムな `kid` を持つトークンをバリデーターに送信し、`refresh_jwks` が最大1回実行され、認可サーバーのキー数が増えないことを確認する。次にフォールバックをrotate-and-mintに意図的に再接続し、偽トークンごとにキー数が増えるのを確認する — 後で再フェッチに戻す。

7. ミックスアップセクションからクライアントサイドのRFC 9207 `iss` チェックを実装する：認可リクエスト前に期待されるissuerを記録し、`iss` が記録したものと一致しない認可レスポンスを拒否する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------------------|
| ASM | "OAuthメタデータドキュメント" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "クライアントメタデータURL" | クライアントIDメタデータドキュメント — `client_id` として使用するHTTPS URL；ASがJSONをプルする。2025年11月25日以降の推奨デフォルト |
| DCR | "セルフサービスのクライアント登録" | RFC 7591 `POST /register` フロー；2025年11月25日に `MAY` フォールバックに降格 |
| JWKS | "JWT検証用の公開鍵" | `jwks_uri` から取得されるJSONウェブキーセット、`kid` でインデックスされる |
| ローテーション vs リフレッシュ | "キーの更新" | *ローテーション* = ASが署名キーを作成/廃止；*リフレッシュ* = リソースサーバーが公開セットを再取得。リソースサーバーはリフレッシュのみ行う |
| リソースインジケーター | "オーディエンスパラメーター" | トークンを1つのサーバーにピン留めするRFC 8707 `resource` パラメーター |
| `aud` クレーム | "オーディエンス" | バリデーターが正規リソースURLと比較するJWTクレーム |
| オーディエンスリプレイ | "トークンリプレイ" | サーバーAに発行されたトークンをサーバーBに提示；オーディエンス検証で防御（仕様：アクセストークン権限制限） |
| Confused deputy | "プロキシトークン誤用" | 静的クライアントIDを持つMCPプロキシがクライアントごとの同意なしにトークンを転送；オーディエンスリプレイとは異なる |
| ミックスアップ攻撃 | "間違ったトークンエンドポイント" | 攻撃者のエンドポイントで正直なASのコードを引き換えるよう誘導されるクライアント；RFC 9207 `iss` でクライアントサイドに防御 |
| `iss` 許可リスト | "信頼された認可サーバー" | 保護リソースメタデータの `authorization_servers` に記述されたセット |
| `resource_metadata` | "PRMドキュメントの場所" | 401/403の `WWW-Authenticate` パラメーターでRFC 9728メタデータURLを指定 |
| パブリッククライアント | "ネイティブまたはブラウザクライアント" | `client_secret` を持たないOAuthクライアント；PKCEが補う |
| `WWW-Authenticate` | "401/403レスポンスヘッダー" | クライアントのリカバリーを駆動する `Bearer error=...` ディレクティブを含む |

## 参考資料

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — このレッスンが実装するMCP認証プロファイル
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025年11月25日の変更内容（CIMD、XAA、DCRの降格）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMDがDCRより優先される根拠
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — ディスカバリーコントラクト
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（フォールバックパス）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — パブリッククライアントの所有証明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — オーディエンスピン留め
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — リソースサーバー発見
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — ミックスアップ攻撃に対して防御する `iss` パラメーター
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 統合されたOAuthの基盤
