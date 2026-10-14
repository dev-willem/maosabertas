//! Todo exemplo de `spec/examples` precisa ser lido pelos tipos Rust.

use maosabertas_protocol::{Utterance, SPEC_VERSION};
use std::{fs, path::Path};

#[test]
fn le_todos_os_exemplos_da_spec() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/examples");
    let mut lidos = 0;
    for entrada in fs::read_dir(&dir).expect("pasta de exemplos") {
        let path = entrada.expect("entrada").path();
        if path.extension() != Some("json".as_ref()) {
            continue;
        }
        let texto = fs::read_to_string(&path).expect("leitura");
        let u: Utterance =
            serde_json::from_str(&texto).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(u.spec_version, SPEC_VERSION, "{}", path.display());

        let volta: Utterance =
            serde_json::from_str(&serde_json::to_string(&u).expect("escrita")).expect("releitura");
        assert_eq!(u, volta, "{}", path.display());
        lidos += 1;
    }
    assert!(lidos > 0, "nenhum exemplo em {}", dir.display());
}
