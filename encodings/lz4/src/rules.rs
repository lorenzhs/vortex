// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::Lz4;

pub(crate) static RULES: ParentRuleSet<Lz4> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Lz4)),
    ParentRuleSet::lift(&CastReduceAdaptor(Lz4)),
]);
